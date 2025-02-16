import torch
import triton
import dataclasses
import argparse
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum, auto

from torch_mqa import torch_mqa
from triton_mqa import triton_mqa

class BenchmarkType(Enum):
    TORCH = auto()
    TRITON = auto()
    TRITON_QK_ATTN = auto()

@dataclasses.dataclass
class BenchmarkConfig:
    batch_size: int
    num_heads: int
    query_length: int
    key_value_length: int
    rope_head_dim: int
    nope_head_dim: int
    kv_lora_rank: int
    attention_mask: Optional[torch.Tensor] = None
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device('cuda')

class Benchmark:
    def __init__(self, torch_mqa_fn, triton_mqa_fn):
        self.torch_mqa = torch_mqa_fn
        self.triton_mqa = triton_mqa_fn

    def setup_tensors(self, config: BenchmarkConfig) -> Tuple[torch.Tensor, ...]:
        scale = (config.rope_head_dim + config.nope_head_dim) ** (-0.5)
        head_dim = config.kv_lora_rank + config.rope_head_dim

        q = torch.randn(
            config.batch_size,
            config.num_heads,
            config.query_length,
            head_dim,
            device=config.device,
            dtype=config.dtype
        )
        k = torch.randn(
            config.batch_size,
            1,
            config.key_value_length,
            head_dim,
            device=config.device,
            dtype=config.dtype
        )
        v = torch.randn(
            config.batch_size,
            1,
            config.key_value_length,
            config.kv_lora_rank,
            device=config.device,
            dtype=config.dtype
        )
        k[..., :config.kv_lora_rank] = v
        return q, k, v, scale

    def run_benchmark(
        self,
        config: BenchmarkConfig,
        benchmark_types: List[BenchmarkType]
    ) -> Dict[str, float]:

        torch.cuda.empty_cache()
        q, k, v, scale = self.setup_tensors(config)
        results = {}

        for bench_type in benchmark_types:
            if bench_type == BenchmarkType.TORCH:
                impl_key = 'torch'
                torch.cuda.reset_peak_memory_stats()
                ms = triton.testing.do_bench(
                    lambda: self.torch_mqa(
                        q, k, v, scale,
                        attention_mask=config.attention_mask
                    )
                )
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated()
                results[impl_key] = ms * 1e3
                results[impl_key + '_mem'] = peak_mem / (1024 ** 2)

            elif bench_type == BenchmarkType.TRITON:
                impl_key = 'triton'
                torch.cuda.reset_peak_memory_stats()
                ms = triton.testing.do_bench(
                    lambda: self.triton_mqa(
                        q, k, v, scale,
                        attention_mask=config.attention_mask,
                        two_stage_decode=False
                    )
                )
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated()
                results[impl_key] = ms * 1e3
                results[impl_key + '_mem'] = peak_mem / (1024 ** 2)

            elif bench_type == BenchmarkType.TRITON_QK_ATTN:
                impl_key = 'triton_qk_attn'
                torch.cuda.reset_peak_memory_stats()
                ms = triton.testing.do_bench(
                    lambda: self.triton_mqa(
                        q, k, v, scale,
                        attention_mask=config.attention_mask,
                        two_stage_decode=True
                    )
                )
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated()
                results[impl_key] = ms * 1e3
                results[impl_key + '_mem'] = peak_mem / (1024 ** 2)

        return results

    def compare_accuracy(self, config: BenchmarkConfig) -> Dict[str, float]:
        q, k, v, scale = self.setup_tensors(config)
        torch_output = self.torch_mqa(q, k, v, scale, attention_mask=config.attention_mask)
        triton_output = self.triton_mqa(q, k, v, scale, attention_mask=config.attention_mask)

        if config.query_length > 1 and config.attention_mask is not None:
            mask = ~config.attention_mask.bool()[:, None, :, None]
            torch_output.masked_fill_(mask, 0.)
            triton_output.masked_fill_(mask, 0.)

        return {
            "max_diff": (torch_output - triton_output).abs().max().item(),
            "mean_diff": (torch_output - triton_output).abs().mean().item()
        }

def create_benchmark_suite(
    x_name: str,
    x_vals: List[Any],
    benchmark_types: List[BenchmarkType],
    fixed_args: Dict[str, Any]
) -> triton.testing.Benchmark:
    return triton.testing.Benchmark(
        x_names=[x_name],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=[t.name.lower() for t in benchmark_types],
        line_names=[t.name.title() for t in benchmark_types],
        styles=[("orange", "-"), ("pink", "-"), ("red", "-"), ("blue", "-"), ("green", "-")][:len(benchmark_types)],
        ylabel="ms",
        plot_name=f"MQA Benchmark - Varying {x_name}",
        args=fixed_args
    )


def generate_random_masks(batch_size, seq_len, sparsity=0.5, device='cuda'):
    """
    Generate random attention masks with a certain fraction (sparsity)
    of positions set to be masked. Here: 0 => masked, 1 => unmasked.
    """
    mask = (torch.rand(batch_size, seq_len, device=device) > sparsity)
    return mask.bool()



def run_all_benchmarks(torch_mqa_fn, triton_mqa_fn):
    
    benchmark = Benchmark(torch_mqa_fn, triton_mqa_fn)

    print("Running all benchmarks...")
    print("\nRunning accuracy comparison...")
    config = BenchmarkConfig(
        batch_size=32,
        num_heads=128,
        query_length=1,
        key_value_length=2048,
        rope_head_dim=64,
        nope_head_dim=128,
        kv_lora_rank=512
    )
    accuracy = benchmark.compare_accuracy(config)
    print(f"Max difference: {accuracy['max_diff']:.6f}")
    print(f"Mean difference: {accuracy['mean_diff']:.6f}")

   
    print("\nRunning prefill benchmark ...")
    prefill_suite = create_benchmark_suite(
        x_name="kv_len",
        x_vals=[128 * i for i in range(1, 17)],
        benchmark_types=[BenchmarkType.TORCH, BenchmarkType.TRITON],
        fixed_args={
            "batch_size": 2,
            "num_heads": 128,
            "rope_head_dim": 64,
            "nope_head_dim": 128,
            "kv_lora_rank": 512
        }
    )

    @triton.testing.perf_report(prefill_suite)
    def prefill_scaling(kv_len, provider, **kwargs):
        cfg = BenchmarkConfig(
            batch_size=kwargs["batch_size"],
            num_heads=kwargs["num_heads"],
            query_length=kv_len,
            key_value_length=kv_len,
            rope_head_dim=kwargs["rope_head_dim"],
            nope_head_dim=kwargs["nope_head_dim"],
            kv_lora_rank=kwargs["kv_lora_rank"]
        )
        return benchmark.run_benchmark(cfg, [BenchmarkType[provider.upper()]])[provider.lower()]

    prefill_scaling.run(show_plots=True, print_data=True)

    
    print("\nRunning decoding batch size scaling benchmark...")
    decode_batch_suite = create_benchmark_suite(
        x_name="batch_size",
        x_vals=[i for i in range(1, 17)],
        benchmark_types=[
            BenchmarkType.TORCH,
            BenchmarkType.TRITON,
            BenchmarkType.TRITON_QK_ATTN
        ],
        fixed_args={
            "key_value_length": 4096,
            "num_heads": 128,
            "rope_head_dim": 64,
            "nope_head_dim": 128,
            "kv_lora_rank": 512
        }
    )

    @triton.testing.perf_report(decode_batch_suite)
    def decode_batch_scaling(batch_size, provider, **kwargs):
        cfg = BenchmarkConfig(
            batch_size=batch_size,
            num_heads=kwargs["num_heads"],
            query_length=1,
            key_value_length=kwargs["key_value_length"],
            rope_head_dim=kwargs["rope_head_dim"],
            nope_head_dim=kwargs["nope_head_dim"],
            kv_lora_rank=kwargs["kv_lora_rank"]
        )
        return benchmark.run_benchmark(cfg, [BenchmarkType[provider.upper()]])[provider.lower()]

    decode_batch_scaling.run(show_plots=True, print_data=True)

    print("\nRunning decoding KV length scaling benchmark...")
    decode_kv_suite = create_benchmark_suite(
        x_name="kv_len",
        x_vals=[512 * i for i in range(1, 9)],
        benchmark_types=[
            BenchmarkType.TORCH,
            BenchmarkType.TRITON,
            BenchmarkType.TRITON_QK_ATTN
        ],
        fixed_args={
            "batch_size": 16,
            "num_heads": 128,
            "rope_head_dim": 64,
            "nope_head_dim": 128,
            "kv_lora_rank": 512
        }
    )

    @triton.testing.perf_report(decode_kv_suite)
    def decode_kv_scaling(kv_len, provider, **kwargs):
        cfg = BenchmarkConfig(
            batch_size=kwargs["batch_size"],
            num_heads=kwargs["num_heads"],
            query_length=1,
            key_value_length=kv_len,
            rope_head_dim=kwargs["rope_head_dim"],
            nope_head_dim=kwargs["nope_head_dim"],
            kv_lora_rank=kwargs["kv_lora_rank"]
        )
        return benchmark.run_benchmark(cfg, [BenchmarkType[provider.upper()]])[provider.lower()]

    decode_kv_scaling.run(show_plots=True, print_data=True)

    print("\nRunning head-dim scaling benchmark...")

    head_dim_suite = create_benchmark_suite(
        x_name="total_head_dim",
        x_vals=[32, 64, 128, 256],
        benchmark_types=[BenchmarkType.TORCH, BenchmarkType.TRITON, BenchmarkType.TRITON_QK_ATTN],
        fixed_args={
            "batch_size": 16,
            "num_heads": 128,
            "query_length": 1,
            "key_value_length": 2048,
            "kv_lora_rank": 512
        }
    )

    @triton.testing.perf_report(head_dim_suite)
    def head_dim_scaling(total_head_dim, provider, **kwargs):
        rope_dim = total_head_dim // 2
        nope_dim = total_head_dim - rope_dim
        cfg = BenchmarkConfig(
            batch_size=kwargs["batch_size"],
            num_heads=kwargs["num_heads"],
            query_length=kwargs["query_length"],
            key_value_length=kwargs["key_value_length"],
            rope_head_dim=rope_dim,
            nope_head_dim=nope_dim,
            kv_lora_rank=kwargs["kv_lora_rank"]
        )
        return benchmark.run_benchmark(cfg, [BenchmarkType[provider.upper()]])[provider.lower()]

    head_dim_scaling.run(show_plots=True, print_data=True)

    print("\nRunning number-of-heads scaling benchmark...")

    num_heads_suite = create_benchmark_suite(
        x_name="num_heads",
        x_vals=[8, 16, 32, 64, 128, 256],
        benchmark_types=[BenchmarkType.TORCH, BenchmarkType.TRITON, BenchmarkType.TRITON_QK_ATTN],
        fixed_args={
            "batch_size": 16,
            "query_length": 1,
            "key_value_length": 2048,
            "rope_head_dim": 64,
            "nope_head_dim": 128,
            "kv_lora_rank": 512
        }
    )

    @triton.testing.perf_report(num_heads_suite)
    def num_heads_scaling(num_heads, provider, **kwargs):
        cfg = BenchmarkConfig(
            batch_size=kwargs["batch_size"],
            num_heads=num_heads,
            query_length=kwargs["query_length"],
            key_value_length=kwargs["key_value_length"],
            rope_head_dim=kwargs["rope_head_dim"],
            nope_head_dim=kwargs["nope_head_dim"],
            kv_lora_rank=kwargs["kv_lora_rank"]
        )
        return benchmark.run_benchmark(cfg, [BenchmarkType[provider.upper()]])[provider.lower()]

    num_heads_scaling.run(show_plots=True, print_data=True)

    print("\nRunning LoRA rank scaling benchmark...")

    lora_rank_suite = create_benchmark_suite(
        x_name="kv_lora_rank",
        x_vals=[64, 128, 256, 512, 1024],
        benchmark_types=[BenchmarkType.TORCH, BenchmarkType.TRITON, BenchmarkType.TRITON_QK_ATTN],
        fixed_args={
            "batch_size": 16,
            "num_heads": 64,
            "query_length": 1,
            "key_value_length": 2048,
            "rope_head_dim": 64,
            "nope_head_dim": 128
        }
    )

    @triton.testing.perf_report(lora_rank_suite)
    def lora_rank_scaling(kv_lora_rank, provider, **kwargs):
        cfg = BenchmarkConfig(
            batch_size=kwargs["batch_size"],
            num_heads=kwargs["num_heads"],
            query_length=kwargs["query_length"],
            key_value_length=kwargs["key_value_length"],
            rope_head_dim=kwargs["rope_head_dim"],
            nope_head_dim=kwargs["nope_head_dim"],
            kv_lora_rank=kv_lora_rank
        )
        return benchmark.run_benchmark(cfg, [BenchmarkType[provider.upper()]])[provider.lower()]

    lora_rank_scaling.run(show_plots=True, print_data=True)

    print("\nRunning mixed dtype benchmark...")

    mixed_dtype_suite = create_benchmark_suite(
        x_name="dtype",
        x_vals=[torch.float16, torch.bfloat16, torch.float32],
        benchmark_types=[BenchmarkType.TORCH, BenchmarkType.TRITON],
        fixed_args={
            "batch_size": 16,
            "num_heads": 128,
            "query_length": 1,
            "key_value_length": 2048,
            "rope_head_dim": 64,
            "nope_head_dim": 128,
            "kv_lora_rank": 512
        }
    )

    @triton.testing.perf_report(mixed_dtype_suite)
    def dtype_scaling(dtype, provider, **kwargs):
        cfg = BenchmarkConfig(
            batch_size=kwargs["batch_size"],
            num_heads=kwargs["num_heads"],
            query_length=kwargs["query_length"],
            key_value_length=kwargs["key_value_length"],
            rope_head_dim=kwargs["rope_head_dim"],
            nope_head_dim=kwargs["nope_head_dim"],
            kv_lora_rank=kwargs["kv_lora_rank"],
            dtype=dtype
        )
        return benchmark.run_benchmark(cfg, [BenchmarkType[provider.upper()]])[provider.lower()]

    dtype_scaling.run(show_plots=True, print_data=True)

    print("\nRunning attention mask density benchmark...")

    mask_density_suite = create_benchmark_suite(
        x_name="mask_density",
        x_vals=[0.0, 0.5, 0.9],
        benchmark_types=[BenchmarkType.TORCH, BenchmarkType.TRITON],
        fixed_args={
            "batch_size": 16,
            "num_heads": 128,
            "rope_head_dim": 64,
            "nope_head_dim": 128,
            "kv_lora_rank": 512,
            "query_length": 128,
            "key_value_length": 128,
        }
    )

    @triton.testing.perf_report(mask_density_suite)
    def mask_density_scaling(mask_density, provider, **kwargs):
        attention_mask = generate_random_masks(
            kwargs["batch_size"], 
            kwargs["query_length"],
            sparsity=mask_density,
            device='cuda'
        )
        cfg = BenchmarkConfig(
            batch_size=kwargs["batch_size"],
            num_heads=kwargs["num_heads"],
            query_length=kwargs["query_length"],
            key_value_length=kwargs["key_value_length"],
            rope_head_dim=kwargs["rope_head_dim"],
            nope_head_dim=kwargs["nope_head_dim"],
            kv_lora_rank=kwargs["kv_lora_rank"],
            attention_mask=attention_mask
        )
        return benchmark.run_benchmark(cfg, [BenchmarkType[provider.upper()]])[provider.lower()]

    mask_density_scaling.run(show_plots=True, print_data=True)


def run_custom_benchmark(torch_mqa_fn, triton_mqa_fn, args):
   
    benchmark = Benchmark(torch_mqa_fn, triton_mqa_fn)

    bench_types = []
    if args.torch:
        bench_types.append(BenchmarkType.TORCH)
    if args.triton:
        bench_types.append(BenchmarkType.TRITON)
    if args.triton_qk_attn:
        bench_types.append(BenchmarkType.TRITON_QK_ATTN)

    if not bench_types:
        bench_types = [BenchmarkType.TORCH, BenchmarkType.TRITON]

    cfg = BenchmarkConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        query_length=args.query_length,
        key_value_length=args.kv_length,
        rope_head_dim=args.rope_head_dim,
        nope_head_dim=args.nope_head_dim,
        kv_lora_rank=args.kv_lora_rank
    )

    if args.accuracy:
        accuracy = benchmark.compare_accuracy(cfg)
        print("\nAccuracy Comparison:")
        print(f"Max difference: {accuracy['max_diff']:.6f}")
        print(f"Mean difference: {accuracy['mean_diff']:.6f}")

    results = benchmark.run_benchmark(cfg, bench_types)
    print("\nPerformance Results:")
    for impl, val in results.items():
        if '_mem' in impl:
            print(f"{impl}: {val:.2f} MB")
        else:
            print(f"{impl}: {val:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="MQA Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")

    parser.add_argument("--torch", action="store_true", help="Run PyTorch implementation")
    parser.add_argument("--triton", action="store_true", help="Run Triton single-kernel decode")
    parser.add_argument("--triton-qk-attn", action="store_true",
                        help="Run Triton QK+Attn multi-pass decode")

    parser.add_argument("--accuracy", action="store_true", help="Compare accuracy")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--query-length", type=int, default=1)
    parser.add_argument("--kv-length", type=int, default=2048)
    parser.add_argument("--rope-head-dim", type=int, default=64)
    parser.add_argument("--nope-head-dim", type=int, default=128)
    parser.add_argument("--kv-lora-rank", type=int, default=512)

    args = parser.parse_args()

    if args.all:
        run_all_benchmarks(torch_mqa, triton_mqa)
    else:
        run_custom_benchmark(torch_mqa, triton_mqa, args)


if __name__ == "__main__":
    main()
