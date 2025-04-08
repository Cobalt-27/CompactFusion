import logging
import time
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_tensor_model_parallel_world_size,
    get_data_parallel_world_size,
)
from xfuser.model_executor.cache.diffusers_adapters import apply_cache_on_transformer

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    DTYPE = torch.half
    engine_config.runtime_config.dtype = DTYPE
    local_rank = get_world_group().local_rank
    text_encoder_2 = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_2", torch_dtype=DTYPE)

    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        logging.info(f"rank {local_rank} quantizing text encoder 2")
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        
    """
    COMPACT
    """
    from xfuser.compact.main import CompactConfig, compact_init, compact_reset, compact_hello
    from xfuser.prof import Profiler, prof_summary, set_torch_profiler
    from xfuser.compact.utils import COMPACT_COMPRESS_TYPE
    COMPACT_METHOD = COMPACT_COMPRESS_TYPE.BINARY
    compact_config = CompactConfig(
        enabled=False,
        compress_func=lambda layer_idx, step: COMPACT_METHOD if step >= 4 else COMPACT_COMPRESS_TYPE.WARMUP,
        sparse_ratio=8,
        comp_rank=16,
        residual=2, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        check_consist=False,
        fastpath=True,
        ref_activation_path='ref_activations',
        dump_activations=False,
        calc_total_error=False,
        low_rank_dim=None,
        delta_decay_factor=0.5
    )
    compact_init(compact_config)
    if compact_config.enable_compress: # IMPORTANT: Compact should be disabled when using pipefusion
        assert args.pipefusion_parallel_degree == 1, "Compact should be disabled when using pipefusion"
    torch.distributed.barrier()

    cache_args = {
            "use_teacache": engine_args.use_teacache,
            "use_fbcache": engine_args.use_fbcache,
            "rel_l1_thresh": 0.6,
            "return_hidden_states_first": False,
            "num_steps": input_config.num_inference_steps,
        }

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        cache_args=cache_args,
        torch_dtype=DTYPE,
        text_encoder_2=text_encoder_2,
    )
    print(pipe)

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)
    
    compact_hello()
    LOOP_COUNT = 1

    for _ in range(LOOP_COUNT):
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        compact_reset()
        Profiler.instance().disable()
        with Profiler.instance().scope("total"):
            output = pipe(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                max_sequence_length=256,
                guidance_scale=0.0,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

        from xfuser.compact.stats import stats_verbose, stats_verbose_steps
        # if local_rank == 0:
        #     stats_verbose()
        #     prof_result = prof_summary(Profiler.instance(), rank=local_rank)
        #     print(str.join("\n", prof_result))

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = f"flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()
