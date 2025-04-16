import logging
import torch
import torch.distributed
import json, os
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state
)
import gc
import time
from xfuser.prof import Profiler
from examples.test_utils import TEST_ENABLE, TEST_MODEL, TEST_METHOD, TEST_LOOP, test_hello
from examples.configs import get_config

_NUM_FID_CANDIDATE = 5000
CFG = 1.5

def flush():
    gc.collect()
    torch.cuda.empty_cache()
from xfuser.compact.main import CompactConfig, compact_init, compact_reset, compact_hello
from xfuser.compact.utils import COMPACT_COMPRESS_TYPE


def customized_compact_config():
    """
    COMPACT
    """
    assert not TEST_ENABLE
    COMPACT_METHOD = COMPACT_COMPRESS_TYPE.BINARY
    compact_config = CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_METHOD if step >= 4 else COMPACT_COMPRESS_TYPE.WARMUP,
        sparse_ratio=8,
        comp_rank=2,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True, 
        simulate=False,
        log_stats=False,
        check_consist=False,
        fastpath=True ,
        ref_activation_path='ref_activations',
        dump_activations=False,
        calc_total_error=False,
        cache_low_rank_dim=None,
        delta_decay_factor=0.3
    )
    return compact_config

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
    if TEST_ENABLE:
        compact_config = get_config(TEST_MODEL, TEST_METHOD)
        _NUM_FID_CANDIDATE = TEST_LOOP
    else:
        compact_config = customized_compact_config()
    compact_init(compact_config)
    if compact_config.enabled: # IMPORTANT: Compact should be disabled when using pipefusion
        assert args.pipefusion_parallel_degree == 1, "Compact should be disabled when using pipefusion"
    torch.distributed.barrier()

    pipe = xFuserFluxPipeline.from_pretrained(
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
    BATCH_NUM = 1
    if TEST_ENABLE:
        BATCH_NUM = 1 # ensure only one prompt for test
    compact_hello()
    if local_rank == 0:
        test_hello()
    total_time = []
    profiler = Profiler().instance()
    profiler.disable()
    for j in range(0, _NUM_FID_CANDIDATE, BATCH_NUM):
        start_time = time.time()
        compact_reset()
        output = pipe(
            height=args.height,
            width=args.width,
            prompt=captions[j:j+BATCH_NUM],
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            max_sequence_length=256,
            guidance_scale=CFG,
            generator=torch.Generator(device='cuda').manual_seed(input_config.seed),
        )
        end_time = time.time()
        total_time.append(end_time - start_time)
        if local_rank == 0:
            print(f'⏱️  time[{j}]: {end_time - start_time:.4f}s')
        if input_config.output_type == 'pil':
            if pipe.is_dp_last_group():
                for k, local_filename in enumerate(filenames[j:j+BATCH_NUM]):
                    output.images[k].save(f'{folder_path}/{j+k:05d}.png')
        flush()
        
    if get_world_group().rank == 0:
        print(f'🎈  avg time: {sum(total_time) / len(total_time):.4f}s')
    get_runtime_state().destory_distributed_env()


if __name__ == '__main__':
    main()
