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

_NUM_FID_CANDIDATE = 5000
CFG = 1.5

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    
from examples.test_utils import TEST_ENABLE, test_hello
from xfuser.compact.utils import COMPACT_COMPRESS_TYPE
from xfuser.compact.patchpara.df_utils import PatchConfig
from xfuser.compact.main import CompactConfig, compact_init, compact_reset, compact_hello

def customized_compact_config():
    """
    COMPACT
    """
    assert not TEST_ENABLE
    prepared_patch_config = PatchConfig(
        use_compact=False,
        async_comm=True,
        async_warmup=2,
    )
    OVERRIDE_WITH_PATCH_PARA = False
    patch_config = prepared_patch_config if OVERRIDE_WITH_PATCH_PARA else None
    COMPACT_METHOD = COMPACT_COMPRESS_TYPE.LOW_RANK_Q
    compact_config = CompactConfig(
        enabled=True,
        override_with_patch_gather_fwd=OVERRIDE_WITH_PATCH_PARA,
        patch_gather_fwd_config=patch_config,
        compress_func=lambda layer_idx, step, tag: (COMPACT_METHOD) if step >= 2 else COMPACT_COMPRESS_TYPE.WARMUP,
        sparse_ratio=8,
        comp_rank=32 if not COMPACT_METHOD == COMPACT_COMPRESS_TYPE.BINARY else -1,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False or COMPACT_METHOD == COMPACT_COMPRESS_TYPE.IDENTITY,
        log_stats=False,
        check_consist=False,
        fastpath=False and COMPACT_METHOD == COMPACT_COMPRESS_TYPE.BINARY,
        ref_activation_path='ref_activations',
        dump_activations=False,
        calc_total_error=False,
        delta_decay_factor=0.5
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
        
    from xfuser.collector.collector import Collector, init
    collector = Collector(
        save_dir="./results/collector", 
        target_steps=None,
        target_layers=None,
        enabled=False,
        rank=local_rank
    )
    init(collector)

    pipe.prepare_run(input_config, steps=1)
    
    with open(args.caption_file, "r") as f:
        captions = json.load(f)
    
    folder_path = args.sample_images_folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # run multiple prompts at a time to save time
    num_prompt_one_step = 1
    compact_hello()
    total_time = []
    profiler = Profiler().instance()
    profiler.disable()
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
        if local_rank == 3:
            print(f"The time used for {j} to {j+num_prompt_one_step}: {end_time - start_time} seconds")
        total_time.append(end_time - start_time)
        if input_config.output_type == 'pil':
            if pipe.is_dp_last_group():
                for k in range(num_prompt_one_step):
                    output.images[k].save(f'{folder_path}/{j+k:05d}.png')
        flush()
        
    if get_world_group().rank == get_world_group().world_size - 1:
        print(f'Average time: {sum(total_time) / len(total_time)} seconds')
    get_runtime_state().destory_distributed_env()


if __name__ == '__main__':
    main()
