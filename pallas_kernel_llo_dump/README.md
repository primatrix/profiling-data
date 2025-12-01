# Pallas RPA Kernel LLO Dump

The data in this directory comes from running benchmarks on the RPA kernel in the `sglang-jax` repository with XLA Mosaic LLO dumping enabled.

## 1. Kernel Source

- Repository: <https://github.com/sgl-project/sglang-jax.git>  
- Kernel: RPA kernel (Pallas-based flash attention kernel)

## 2. LLO Dump Environment Variable

Before running the benchmark, enable XLA Mosaic LLO dumping with:

```bash
export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=/tmp/mosaic_dumps"
```

This causes Mosaic/XLA to dump intermediate artifacts (including LLO) into `/tmp/mosaic_dumps`. The LLO files relevant to this kernel were extracted from that directory into the current one.

## 3. Kernel Execution Configuration

The RPA kernel was benchmarked with the following configuration (prefill mode only):

```python
bench_modes = ["prefill"]
page_size_config = [128]
max_num_batched_tokens_config_for_decode = [
    128,
]
max_num_batched_tokens_config_for_prefill = [
    8192,
]
q_head_num_config = [32]
kv_head_num_config = [8]
head_dim_config = [128]
max_kv_cache_tokens_config = [600000]
all_combinations = []
config_of_modes = {}
max_context_len = 40960
```

## 4. Command Used

From the root of the `sglang-jax` repository, the benchmark was launched with:

```bash
python3 benchmark/kernels/flash_attention/bench_flashattention.py
```

In summary, the LLO dump files in this directory are produced by running the above RPA kernel benchmark with the specified environment variable and configuration, using XLA Mosaic’s dump mechanism.
