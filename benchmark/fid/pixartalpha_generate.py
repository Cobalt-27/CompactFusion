import logging
import torch
import torch.distributed
import json, os
from xfuser import xFuserPixArtAlphaPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state
)
import gc
import time


_NUM_FID_CANDIDATE = 5000
CFG = 1.5

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    parser = FlexibleArgumentParser(description='xFuser Arguments')
    parser.add_argument('--caption_file', type=str, default='captions_coco.json')
    parser.add_argument('--sample_images_folder', type=str, default='sample_images')
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    DTYPE = torch.half
    engine_config.runtime_config.dtype = DTYPE
    local_rank = get_world_group().local_rank
    
    """
    COMPACT
    """
    from xfuser.compact.main import CompactConfig, compact_init, compact_reset, compact_hello
    from xfuser.compact.utils import COMPACT_COMPRESS_TYPE
    COMPACT_METHOD = COMPACT_COMPRESS_TYPE.BINARY
    compact_config = CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_METHOD if step >= 2 else COMPACT_COMPRESS_TYPE.WARMUP,
        sparse_ratio=8,
        comp_rank=16,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True, 
        simulate=True,
        log_stats=True,
        check_consist=False,
        fastpath=False,
        ref_activation_path='ref_activations',
        dump_activations=False,
        calc_total_error=False,
        low_rank_dim=None,
        delta_decay_factor=0.3
    )
    compact_init(compact_config)
    if compact_config.enable_compress: # IMPORTANT: Compact should be disabled when using pipefusion
        assert args.pipefusion_parallel_degree == 1, "Compact should be disabled when using pipefusion"
    torch.distributed.barrier()

    pipe = xFuserPixArtAlphaPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=DTYPE,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f'rank {local_rank} sequential CPU offload enabled')
    else:
        pipe = pipe.to(f'cuda:{local_rank}')

    pipe.prepare_run(input_config, steps=1)
    
    from dataloader import get_dataset
    with open(args.caption_file, "r") as f:
        captions = json.load(f)
    dataset = get_dataset()
    filenames = dataset["filename"][:len(captions)]
    
    folder_path = args.sample_images_folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # run multiple prompts at a time to save time
    num_prompt_one_step = 120
    compact_hello()
    for j in range(0, _NUM_FID_CANDIDATE, num_prompt_one_step):
        start_time = time.time()
        compact_reset()
        output = pipe(
            height=args.height,
            width=args.width,
            prompt=captions[j:j+num_prompt_one_step],
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            max_sequence_length=256,
            guidance_scale=CFG,
            generator=torch.Generator(device='cuda').manual_seed(input_config.seed),
        )
        end_time = time.time()
        print(f'Time taken: {end_time - start_time} seconds')
        if input_config.output_type == 'pil':
            if pipe.is_dp_last_group():
                for k, local_filename in enumerate(filenames[j:j+num_prompt_one_step]):
                    output.images[k].save(f'{folder_path}/{local_filename}')
        print(f'{j}-{j+num_prompt_one_step-1} generation finished!')
        flush()

    get_runtime_state().destory_distributed_env()


if __name__ == '__main__':
    main()
