# Tiny Latent Reasoning

Reproductions of latent visual reasoning methods. The first milestone is checkpoint evaluation with SGLang and a pinned lmms-eval submodule.

| Paper | Baseline sizes | Evaluation coverage |
| --- | --- | --- |
| [Latent Visual Reasoning](https://arxiv.org/html/2509.24251v2) | 3B, 7B | V* overall and both subsets, MMVP, and the BLINK subsets used in Tables 1 and 2, including IQ-Test |
| [HyLaR](https://arxiv.org/html/2604.20328v2) | 3B, 7B | V*, HRBench 4K and 8K, MMVP, MMStar, SEED-Bench-2-Plus, BLINK, HallusionBench |
| [LUT](https://arxiv.org/html/2608.00743v1) | 7B | V*, HRBench 4K and 8K, MME-RealWorld-Lite, MMStar, BLINK, VisualPuzzles |

## Baseline evaluation

On Linux x86-64 with a CUDA GPU, initialize the submodule and supply a downloaded Qwen2.5-VL-Instruct snapshot matching the requested size.

```sh
git submodule update --init
uv run --frozen --extra eval scripts/baseline.py lvr --size 3b \
    --checkpoint /path/to/snapshot --output /path/to/results
```

Use `--limit 1` for a smoke run and `--batch-size` to tune throughput. The runner starts the SGLang engine through lmms-eval. Exact task IDs for each paper and model size are in [the registry](recipes/benchmarks.toml). The VisualPuzzles task uses the full original split and per-question accuracy. The upstream UniG2U configurations use sampled categories. Prompt and scorer equivalence to LUT and paper-specific metric selection for HallusionBench and MMVP still require validation.

## Development

Install pre-commit and run `pre-commit install` before contributing.

Run the registry and scoring regression checks with `uv run --frozen --extra eval python -m unittest discover -s tests`. The pinned MMVP task exposes question and pair accuracy and uses corrected labels for four questions. Record the metric and label revision when comparing paper scores.
