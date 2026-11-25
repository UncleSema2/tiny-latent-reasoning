# Tiny Latent Reasoning

Reproductions of latent visual reasoning methods. The first milestone is checkpoint evaluation with SGLang and a pinned lmms-eval submodule.

| Paper | Baseline sizes | Evaluation coverage |
| --- | --- | --- |
| [Latent Visual Reasoning](https://arxiv.org/html/2509.24251v2) | 3B, 7B | V* overall and both subsets, MMVP, and the BLINK subsets used in Tables 1 and 2, including IQ-Test |
| [HyLaR](https://arxiv.org/html/2604.20328v2) | 3B, 7B | V*, HRBench 4K and 8K, MMVP, MMStar, SEED-Bench-2-Plus, BLINK, HallusionBench |
| [LUT](https://arxiv.org/html/2608.00743v1) | 7B | V*, HRBench 4K and 8K, MME-RealWorld-Lite, MMStar, BLINK, VisualPuzzles |

## Benchmark results

### Qwen2.5-VL-3B-Instruct

Full evaluation of the base checkpoint on the LVR 3B suite. Scores are percentages. The last column subtracts the published Qwen2.5-VL-3B score from our score in percentage points. A dash denotes an unavailable value. HyLaR reports a single MMVP score without specifying question or pair aggregation in Table 4. The † comparison assumes question accuracy. [HyLaR, Table 4](https://arxiv.org/html/2604.20328v2#S4.T4)

| Benchmark and metric | Evaluated | Ours (%) | HyLaR paper, base 3B (%) | Difference (pp) |
| --- | ---: | ---: | ---: | ---: |
| V* overall | 191 | **76.96** | 70.68 | +6.28 |
| V* Direct Attributes | 115 | **84.35** | — | — |
| V* Relative Position | 76 | **64.47** | — | — |
| MMVP question accuracy | 300 | **68.00** | 62.00† | +6.00† |
| MMVP pair accuracy | 150 pairs | **40.00** | — | — |
| BLINK IQ-Test | 150 | **24.00** | — | — |
| BLINK Jigsaw | 150 | **48.67** | — | — |
| HRBench-4K overall | — | — | 61.75 | — |
| HRBench-8K overall | — | — | 60.25 | — |
| MMStar | — | — | 52.40 | — |
| SEED-Bench-2-Plus | — | — | 58.82 | — |

The two V* subsets partition the 191 overall questions. Our run used batch size 8 and the LVR image budget of 128–5,120 visual tokens. The checkpoint revision was `66285546d2b821cf421d4f5eb2576359d3770cd3` and the lmms-eval revision was `9134731f4f617e62ee80bcae9d0d5d9695ea55ef`. The harness recorded 2,478 generated tokens in 199.82 seconds (12.40 tokens/s). The cause of the difference from HyLaR remains unverified. [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3)

LVR Table 2 reports trained 3B variants at 4, 8 and 16 latent steps and omits the base Qwen row. For example, its 67.0 on V* and 58.0 on MMVP are scores for LVR after RL at 8 and 16 steps respectively. LUT evaluates 7B models. [LVR, Tables 1–2](https://arxiv.org/html/2509.24251v2#S4.T2) · [LUT, Section 4](https://arxiv.org/html/2608.00743v1#S4)

### Published Qwen2.5-VL-7B baselines

Each column gives the base model score as reported by that paper. A dash means the main tables omit that metric for the base model. The project has completed the 3B run above; local 7B results are pending. [LVR, Table 1](https://arxiv.org/html/2509.24251v2#S4.T1) · [HyLaR, Tables 1–2](https://arxiv.org/html/2604.20328v2#S4.T1) · [LUT, Tables 1–3](https://arxiv.org/html/2608.00743v1#S4.T1)

| Benchmark and metric (%) | LVR | HyLaR | LUT |
| --- | ---: | ---: | ---: |
| V* overall | 78.5 | 76.44 | 76.44 |
| V* Direct Attributes | 81.7 | 77.39 | 77.39 |
| V* Relative Position | 73.7 | 75.00 | 75.00 |
| MMVP (paper score) | 66.7 | 65.67 | — |
| BLINK overall | — | 53.60 | 53.60 |
| BLINK Counting | 66.7 | — | — |
| BLINK IQ-Test | 26.0 | — | — |
| BLINK Jigsaw | 52.0 | — | — |
| BLINK Relative Reflectance | 38.8 | — | — |
| BLINK Spatial Relation | 87.4 | — | — |
| HRBench-4K overall | — | 68.00 | 67.00 |
| HRBench-4K single | — | 80.25 | 83.50 |
| HRBench-4K cross | — | 55.75 | 50.50 |
| HRBench-8K overall | — | 63.75 | 64.38 |
| HRBench-8K single | — | 73.75 | 76.50 |
| HRBench-8K cross | — | 53.75 | 52.25 |
| MMStar | — | 59.70 | 59.70 |
| SEED-Bench-2-Plus | — | 65.31 | — |
| HallusionBench | — | 56.57 | — |
| MME-RealWorld-Lite overall | — | — | 43.67 |
| MME-RealWorld-Lite reasoning | — | — | 36.00 |
| MME-RealWorld-Lite perception | — | — | 48.59 |
| VisualPuzzles | — | — | 32.71 |
| Mean of V*, HRBench-4K, HRBench-8K and MME-RealWorld-Lite | — | — | 62.87 |

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
