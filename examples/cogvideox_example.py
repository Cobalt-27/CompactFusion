import logging
import time
import torch
import torch.distributed
from diffusers import AutoencoderKLTemporalDecoder
from xfuser import xFuserCogVideoXPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)
from diffusers.utils import export_to_video
from xfuser.compact.main import CompactConfig, compact_init, compact_reset, compact_hello
from xfuser.compact.utils import COMPACT_COMPRESS_TYPE
from xfuser.compact.patchpara.df_utils import PatchConfig
from xfuser.prof import Profiler, prof_summary

def customized_compact_config():
    """
    COMPACT configuration for CogVideoX
    """
    prepared_patch_config = PatchConfig(
        use_compact=False,
        async_comm=True,
        async_warmup=1,
    )
    OVERRIDE_WITH_PATCH_PARA = False
    patch_config = prepared_patch_config if OVERRIDE_WITH_PATCH_PARA else None
    COMPACT_METHOD = COMPACT_COMPRESS_TYPE.INT2
    compact_config = CompactConfig(
        enabled=True,
        override_with_patch_gather_fwd=OVERRIDE_WITH_PATCH_PARA,
        patch_gather_fwd_config=patch_config,
        compress_func=lambda layer_idx, step: (COMPACT_METHOD) if step >= 1 else COMPACT_COMPRESS_TYPE.WARMUP,
        sparse_ratio=8,
        comp_rank=32 if not COMPACT_METHOD in [COMPACT_COMPRESS_TYPE.BINARY, COMPACT_COMPRESS_TYPE.INT2] else -1,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False or COMPACT_METHOD == COMPACT_COMPRESS_TYPE.IDENTITY,
        log_stats=True,
        check_consist=False,
        fastpath=True and COMPACT_METHOD in [COMPACT_COMPRESS_TYPE.BINARY, COMPACT_COMPRESS_TYPE.INT2],
        delta_decay_factor=0.5
    )
    return compact_config

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for CogVideo"

    # Initialize Compact
    """
    Compact
    """
    from configs import get_config
    compact_config = get_config("CogVideoX", "ring")
    if compact_config.enabled:
        assert args.pipefusion_parallel_degree == 1, "Compact should be disabled when using pipefusion"
    compact_init(compact_config)
    torch.distributed.barrier()

    pipe = xFuserCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    if args.enable_tiling:
        pipe.vae.enable_tiling()

    if args.enable_slicing:
        pipe.vae.enable_slicing()
        
    
    """
    Collector
    """
    from xfuser.collector.collector import Collector, init
    collector = Collector(
        save_dir="./results/collector", 
        target_steps=None,
        target_layers=None,
        enabled=False,
        rank=local_rank
    )
    init(collector)

    # warmup
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=1,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    ).frames[0]

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # Reset Compact before inference
    compact_reset()
    Profiler.instance().reset()
    with Profiler.instance().scope("total"):
        output = pipe(
            height=input_config.height,
            width=input_config.width,
            num_frames=input_config.num_frames,
            prompt=input_config.prompt,
            num_inference_steps=input_config.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        ).frames[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if is_dp_last_group():
        resolution = f"{input_config.width}x{input_config.height}"
        output_filename = f"results/cogvideox_{parallel_info}_{resolution}.mp4"
        export_to_video(output, output_filename, fps=8)
        print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
        # Print profiler summary
        prof_result = prof_summary(Profiler.instance(), rank=local_rank)
        print(str.join("\n", prof_result))
    
    # Sync profiler
    Profiler.instance().sync()
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
