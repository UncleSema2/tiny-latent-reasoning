# Tiny Latent Reasoning

Reproductions of latent visual reasoning methods. The first milestone is checkpoint evaluation with SGLang and a pinned lmms-eval submodule.

| Paper | Baseline sizes | Evaluation coverage |
| --- | --- | --- |
| [Latent Visual Reasoning](https://arxiv.org/html/2509.24251v2) | 3B, 7B | V* overall and both subsets, MMVP, and the BLINK subsets used in Tables 1 and 2, including IQ-Test |
| [HyLaR](https://arxiv.org/html/2604.20328v2) | 3B, 7B | V*, HRBench 4K and 8K, MMVP, MMStar, SEED-Bench-2-Plus, BLINK, HallusionBench |
| [LUT](https://arxiv.org/html/2608.00743v1) | 7B | V*, HRBench 4K and 8K, MME-RealWorld-Lite, MMStar, BLINK, VisualPuzzles |

## Benchmark results

### Qwen2.5-VL-3B-Instruct

Full evaluation of the base checkpoint on the LVR 3B suite. Scores are percentages. The last column subtracts the published Qwen2.5-VL-3B score from our score in percentage points. A dash denotes an unavailable value. HyLaR reports a single MMVP score. Its released judging code aggregates per-question decisions. [HyLaR, Table 4](https://arxiv.org/html/2604.20328v2#S4.T4)

| Benchmark and metric | Evaluated | Ours (%) | HyLaR paper, base 3B (%) | Difference (pp) |
| --- | ---: | ---: | ---: | ---: |
| V* overall | 191 | **75.39** | 70.68 | +4.71 |
| V* Direct Attributes | 115 | **84.35** | — | — |
| V* Relative Position | 76 | **61.84** | — | — |
| MMVP question accuracy | 300 | **64.00** | 62.00 | +2.00 |
| MMVP pair accuracy | 150 pairs | **34.00** | — | — |
| BLINK IQ-Test | 150 | **26.67** | — | — |
| BLINK Jigsaw | 150 | **38.67** | — | — |
| HRBench-4K overall | — | — | 61.75 | — |
| HRBench-8K overall | — | — | 60.25 | — |
| MMStar | — | — | 52.40 | — |
| SEED-Bench-2-Plus | — | — | 58.82 | — |

The two V* subsets partition the 191 overall questions. Our run used batch size 16 and the LVR image budget of 128–5,120 visual tokens. The checkpoint revision was `66285546d2b821cf421d4f5eb2576359d3770cd3` and the lmms-eval revision was `9134731f4f617e62ee80bcae9d0d5d9695ea55ef`. The harness recorded 2,480 generated tokens in 162.52 seconds (15.26 tokens/s). The cause of the difference from HyLaR remains unverified. [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3)

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

On Linux x86-64 with a CUDA GPU, clone with `--recurse-submodules` or run `git submodule update --init` once. Run a full baseline suite with one command:

```sh
uv run --frozen --extra eval scripts/baseline.py lvr --size 3b --batch-size 16 --output results/lvr-3b
```

Choose `lvr`, `hylar`, or `lut` and a model size from the coverage table. The runner downloads the pinned Qwen checkpoint, starts the SGLang engine and saves aggregate scores and per-question predictions. Set `HF_HOME` and `LMMS_EVAL_DATASETS_CACHE` to storage with room for model weights, source data and decoded datasets. Use `--checkpoint` for a local snapshot, `--limit 1` for a smoke run and `--batch-size` to tune throughput. [Paper suites](recipes/benchmarks.toml) · [Dataset revisions](recipes/datasets.toml)

### Evaluation protocols

- **Visual inputs.** LVR uses 128–5,120 visual tokens per image. HyLaR uses 128–16,384, following its released inference example for the trained model. For these suites the runner sets the size limits on the SGLang image processor. LUT uses the checkpoint processor defaults. [LVR experimental settings](https://arxiv.org/html/2509.24251v2#S4) · [HyLaR inference command](https://github.com/EthenCheng/HyLaR/blob/8105a5ba032ae2373ff864aeeeb1f253287cac0b/Evaluate/run_eval_HyLar.sh)
- **Prompts and scoring.** The HyLaR evaluation code reads prepared question text, allows up to 1,024 output tokens in its launcher and provides a separate per-question semantic judge (default GPT-5). Here lmms-eval builds task-specific prompts and extracts answers locally. The released scripts do not identify the exact checkpoint revision, prompts and label files used for the paper's base-model rows. Differences in scores therefore cannot be attributed to a single cause. [HyLaR inference](https://github.com/EthenCheng/HyLaR/blob/8105a5ba032ae2373ff864aeeeb1f253287cac0b/Evaluate/eval_HyLar.py) · [HyLaR judge](https://github.com/EthenCheng/HyLaR/blob/8105a5ba032ae2373ff864aeeeb1f253287cac0b/Evaluate/Accuracy_judge.py)
- **MMVP and HRBench.** MMVP reports both question and pair accuracy and includes four corrected labels. HRBench uses the harness's option extractor, assigns zero credit to unrecognized answers, and averages accuracy over cycle categories. [MMVP implementation](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/9134731f4f617e62ee80bcae9d0d5d9695ea55ef/lmms_eval/tasks/mmvp/utils.py) · [HRBench implementation](tasks/hrbench/score.py)
- **HallusionBench and VisualPuzzles.** HallusionBench uses the 951 image questions and the harness's built-in local yes/no scorer. It reports answer, question-group and figure-group accuracy. VisualPuzzles uses all 1,168 questions from the original split with per-question accuracy. [HallusionBench task](tasks/hallusion/hallusion_local.yaml) · [VisualPuzzles task](tasks/visualpuzzles/visualpuzzles.yaml)

## Development

Install pre-commit and run `pre-commit install` before contributing.

Run the registry and scoring regression checks with `uv run --frozen --extra eval python -m unittest discover -s tests`.
