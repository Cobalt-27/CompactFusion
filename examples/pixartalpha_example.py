import time
import os
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserPixArtAlphaPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_world_size,
    get_runtime_state,
    get_data_parallel_rank,
)


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    text_encoder = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder", torch_dtype=torch.float16)
    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        print(f"rank {local_rank} quantizing text encoder")
        quantize(text_encoder, weights=qfloat8)
        freeze(text_encoder)

    """
    COMPACT
    """
    from xfuser.compact.main import CompactConfig, compact_init, compact_reset, compact_hello
    from xfuser.prof import Profiler, prof_summary, set_torch_profiler
    from xfuser.compact.utils import COMPACT_COMPRESS_TYPE
    COMPACT_METHOD = COMPACT_COMPRESS_TYPE.BINARY
    compact_config = CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_METHOD if step >= 2 else COMPACT_COMPRESS_TYPE.WARMUP,
        sparse_ratio=8,
        comp_rank=16,
        residual=2, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=True,
        check_consist=False,
        fastpath=False,
        ref_activation_path='ref_activations',
        dump_activations=False,
        calc_total_error=False,
        delta_decay_factor=0.5,
        quantized_cache=False,
        low_rank_dim=16
    )
    compact_init(compact_config)
    if compact_config.enable_compress: # IMPORTANT: Compact should be disabled when using pipefusion
        assert args.pipefusion_parallel_degree == 1, "Compact should be disabled when using pipefusion"
    torch.distributed.barrier()

    pipe = xFuserPixArtAlphaPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
        text_encoder=text_encoder,
    ).to(f"cuda:{local_rank}")
    model_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    pipe.prepare_run(input_config)
    
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    compact_hello()
    LOOP_COUNT = 1

    for _ in range(LOOP_COUNT):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        compact_reset()
        Profiler.instance().reset()
        with Profiler.instance().scope("total"):
            output = pipe(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                use_resolution_binning=input_config.use_resolution_binning,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
        from xfuser.compact.stats import stats_verbose, stats_verbose_steps
        if local_rank == 0:
            stats_verbose()
            # stats_verbose_steps(keys=["8-0-k"])
            prof_result = prof_summary(Profiler.instance(), rank=local_rank)
            print(str.join("\n", prof_result))

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}_tc_{engine_args.use_torch_compile}"
    )
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            if not os.path.exists("results"):
                os.mkdir("results")
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                img_file = (
                    f"./results/pixart_alpha_result_{parallel_info}_{image_rank}.png"
                )
                image.save(img_file)
                print(img_file)

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, model memory: {model_memory/1e9:.2f} GB, overall memory: {peak_memory/1e9:.2f} GB"
        )
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
