# Tiny Latent Reasoning

Reproductions of latent visual reasoning methods. The first milestone is checkpoint evaluation with SGLang and a pinned lmms-eval submodule.

| Paper | Baseline sizes | Evaluation coverage |
| --- | --- | --- |
| [Latent Visual Reasoning](https://arxiv.org/html/2509.24251v2) | 3B, 7B | V* overall and both subsets, MMVP, and the BLINK subsets used in Tables 1 and 2, including IQ-Test |
| [HyLaR](https://arxiv.org/html/2604.20328v2) | 3B, 7B | V*, HRBench 4K and 8K, MMVP, MMStar, SEED-Bench-2-Plus, BLINK, HallusionBench |
| [LUT](https://arxiv.org/html/2608.00743v1) | 7B | V*, HRBench 4K and 8K, MME-RealWorld-Lite, MMStar, BLINK, VisualPuzzles |

## Reproduction plan

1. **Establish the baseline.** Pin checkpoints, data and the evaluation harness. Run each base model on its paper's full suite and compare the same splits and scoring conventions.
2. **Implement each method.** Give LVR, HyLaR and LUT separate training recipes with pinned inputs, latent-step settings, seeds and checkpoint metadata. Validate each recipe on a small run before scaling training.
3. **Evaluate trained checkpoints.** Run each method on its original suite, record accuracy and inference cost alongside the base model, and expose training and evaluation through one command per paper. Re-run the existing recipe checks when adding a method.

## Benchmark results

Scores are percentages except for mean output tokens. Every local column comes from a complete run of the base Qwen2.5-VL-Instruct checkpoint on the named suite. A dash marks an unmeasured or unpublished entry. V* overall includes both subsets. When a suite runs only V* overall, the subset scores use the category on each scored prediction. Each MMVP pair contains two questions.

### Qwen2.5-VL-3B-Instruct

HyLaR Table 4 reports six base-model scores. Differences below subtract the published value from our HyLaR-suite score in percentage points. We compare MMVP question accuracy with the paper's MMVP column because the released judge scores individual questions. See Evaluation protocols for differences in image processing, prompts and scoring. [HyLaR, Table 4](https://arxiv.org/html/2604.20328v2#S4.T4)

| Benchmark and metric | Examples or pairs | LVR suite (ours) | HyLaR suite (ours) | HyLaR paper | HyLaR Δ (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| V* overall | 191 | 75.39 | 76.44 | 70.68 | +5.76 |
| V* Direct Attributes | 115 | 84.35 | 84.35 | — | — |
| V* Relative Position | 76 | 61.84 | 64.47 | — | — |
| MMVP question accuracy | 300 | 64.00 | 65.00 | 62.00 | +3.00 |
| MMVP pair accuracy | 150 | 34.00 | 34.67 | — | — |
| BLINK IQ-Test | 150 | 26.67 | — | — | — |
| BLINK Jigsaw | 150 | 38.67 | — | — | — |
| HRBench-4K overall | 800 | — | 67.00 | 61.75 | +5.25 |
| HRBench-8K overall | 800 | — | 63.50 | 60.25 | +3.25 |
| MMStar | 1500 | — | 55.60 | 52.40 | +3.20 |
| SEED-Bench-2-Plus | 2277 | — | 67.90 | 58.82 | +9.08 |
| HRBench-4K cross | 400 | — | 49.75 | — | — |
| HRBench-4K mean output tokens | 800 | — | 2.24 | — | — |
| HRBench-4K single | 400 | — | 84.25 | — | — |
| HRBench-8K cross | 400 | — | 46.50 | — | — |
| HRBench-8K mean output tokens | 800 | — | 2.18 | — | — |
| HRBench-8K single | 400 | — | 80.50 | — | — |
| MMStar · coarse perception | 250 | — | 68.80 | — | — |
| MMStar · fine-grained perception | 250 | — | 48.40 | — | — |
| MMStar · instance reasoning | 250 | — | 61.20 | — | — |
| MMStar · logical reasoning | 250 | — | 55.20 | — | — |
| MMStar · math | 250 | — | 60.40 | — | — |
| MMStar · science & technology | 250 | — | 39.60 | — | — |
| SEED-Bench-2-Plus · Chart | 810 | — | 64.32 | — | — |
| SEED-Bench-2-Plus · Map | 807 | — | 59.23 | — | — |
| SEED-Bench-2-Plus · Web | 660 | — | 82.88 | — | — |
| V* mean output tokens | 191 | 2.58 | 2.57 | — | — |

The LVR paper reports trained 3B models at 4, 8 and 16 latent steps. For example, its 67.0 on V* and 58.0 on MMVP are scores after RL at 8 and 16 steps respectively. LUT evaluates 7B models. [LVR, Table 2](https://arxiv.org/html/2509.24251v2#S4.T2) · [LUT, Section 4](https://arxiv.org/html/2608.00743v1#S4)

### Qwen2.5-VL-7B-Instruct

Each suite has a local and a published column. The local HallusionBench score uses answer accuracy on the image split. HyLaR gives one HallusionBench score without specifying that scorer. [LVR, Table 1](https://arxiv.org/html/2509.24251v2#S4.T1) · [HyLaR, Tables 1–2](https://arxiv.org/html/2604.20328v2#S4.T1) · [LUT, Tables 1–3](https://arxiv.org/html/2608.00743v1#S4.T1)

| Benchmark and metric | LVR ours | LVR paper | HyLaR ours | HyLaR paper | LUT ours | LUT paper |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V* overall | 81.68 | 78.5 | 81.68 | 76.44 | 82.20 | 76.44 |
| V* Direct Attributes | 84.35 | 81.7 | 84.35 | 77.39 | 85.22 | 77.39 |
| V* Relative Position | 77.63 | 73.7 | 77.63 | 75.00 | 77.63 | 75.00 |
| MMVP question accuracy | 75.00 | 66.7 | 75.00 | 65.67 | — | — |
| BLINK overall | — | — | 55.97 | 53.60 | 55.31 | 53.60 |
| BLINK Counting | 68.33 | 66.7 | 69.17 | — | 68.33 | — |
| BLINK IQ-Test | 26.67 | 26.0 | 26.67 | — | 24.67 | — |
| BLINK Jigsaw | 68.00 | 52.0 | 69.33 | — | 60.67 | — |
| BLINK Relative Reflectance | 34.33 | 38.8 | 34.33 | — | 34.33 | — |
| BLINK Spatial Relation | 89.51 | 87.4 | 88.81 | — | 88.81 | — |
| HRBench-4K overall | — | — | 71.50 | 68.00 | 71.38 | 67.00 |
| HRBench-4K single | — | — | 89.00 | 80.25 | 89.25 | 83.50 |
| HRBench-4K cross | — | — | 54.00 | 55.75 | 53.50 | 50.50 |
| HRBench-8K overall | — | — | 68.50 | 63.75 | 68.88 | 64.38 |
| HRBench-8K single | — | — | 83.00 | 73.75 | 83.25 | 76.50 |
| HRBench-8K cross | — | — | 54.00 | 53.75 | 54.50 | 52.25 |
| MMStar | — | — | 62.60 | 59.70 | 61.93 | 59.70 |
| SEED-Bench-2-Plus | — | — | 70.88 | 65.31 | — | — |
| HallusionBench answer accuracy | — | — | 68.45 | 56.57 | — | — |
| MME-RealWorld-Lite overall | — | — | — | — | 45.39 | 43.67 |
| MME-RealWorld-Lite reasoning | — | — | — | — | 38.13 | 36.00 |
| MME-RealWorld-Lite perception | — | — | — | — | 50.04 | 48.59 |
| VisualPuzzles | — | — | — | — | 32.96 | 32.71 |
| Mean of V*, HRBench-4K, HRBench-8K and MME-RealWorld-Lite | — | — | — | — | 66.96 | 62.87 |
| BLINK Art Style | — | — | 67.52 | — | 66.67 | — |
| BLINK Forensic Detection | — | — | 45.45 | — | 45.45 | — |
| BLINK Functional Correspondence | — | — | 26.92 | — | 26.15 | — |
| BLINK Multi View Reasoning | — | — | 53.38 | — | 53.38 | — |
| BLINK Object Localization | — | — | 49.18 | — | 50.82 | — |
| BLINK Relative Depth | — | — | 75.81 | — | 77.42 | — |
| BLINK Semantic Correspondence | — | — | 33.09 | — | 33.81 | — |
| BLINK Visual Correspondence | — | — | 53.49 | — | 53.49 | — |
| BLINK Visual Similarity | — | — | 90.37 | — | 90.37 | — |
| HRBench-4K mean output tokens | — | — | 2.02 | — | 2.02 | — |
| HRBench-8K mean output tokens | — | — | 2.02 | — | 2.02 | — |
| HallusionBench figure-group accuracy | — | — | 37.86 | — | — | — |
| HallusionBench question-group accuracy | — | — | 41.76 | — | — | — |
| MME-RealWorld-Lite mean output tokens | — | — | — | — | 2.00 | — |
| MMStar · coarse perception | — | — | 70.80 | — | 70.40 | — |
| MMStar · fine-grained perception | — | — | 54.80 | — | 54.40 | — |
| MMStar · instance reasoning | — | — | 72.00 | — | 70.40 | — |
| MMStar · logical reasoning | — | — | 65.20 | — | 64.80 | — |
| MMStar · math | — | — | 66.80 | — | 67.60 | — |
| MMStar · science & technology | — | — | 46.00 | — | 44.00 | — |
| MMVP pair accuracy | 51.33 | — | 51.33 | — | — | — |
| SEED-Bench-2-Plus · Chart | — | — | 68.77 | — | — | — |
| SEED-Bench-2-Plus · Map | — | — | 61.96 | — | — | — |
| SEED-Bench-2-Plus · Web | — | — | 84.39 | — | — | — |
| V* mean output tokens | 2.63 | — | 2.63 | — | 2.63 | — |

On V*, our 7B LVR run scores 81.68% against 78.5% in the paper. BLINK Jigsaw shows a larger difference (68.00% versus 52.0%). For LUT, the mean across the four perception benchmarks is 66.96% against 62.87%. The 7B papers label their MMVP column simply MMVP. Both local scoring conventions are shown. LUT cites public reports as the source of its Qwen baseline. [LUT, Section 4.1](https://arxiv.org/html/2608.00743v1#S4.SS1)

### Inference throughput

Measured on one A100 80 GB. Seconds include the SGLang generation calls measured by lmms-eval. Image sizes and output lengths vary by suite.

| Suite | Model size | Batch size | Requests | Output tokens | Seconds | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LVR | 3B | 16 | 982 | 2480 | 162.52 | 15.26 |
| HyLaR | 3B | 16 | 5868 | 13186 | 2344.72 | 5.62 |
| LVR | 7B | 16 | 1379 | 3456 | 337.03 | 10.25 |
| HyLaR | 7B | 16 | 8911 | 72734 | 6451.81 | 11.27 |
| LUT | 7B | 8 | 8279 | 19164 | 8591.15 | 2.23 |

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

## Released LVR checkpoint

Evaluate the published 7B weights on the LVR suite with fixed-step decoding:

```sh
uv run --frozen --extra eval scripts/baseline.py lvr --lvr-steps 8 --batch-size 16 --output results/lvr-7b-8
```

Choose 4, 8 or 16 steps. The first decode of the start marker uses its token embedding. Each subsequent latent step takes the previous normalized hidden state as its input. The budget is tracked separately for each request. Evaluation uses greedy decoding with a 512-token limit and the same image limits, prompts and scoring as the local baseline. [Reference decoder](https://github.com/VincentLeebang/lvr/blob/637a9f7e17e865b1372b65f182197fbc4a5a1449/src/model/qwen_lvr_model.py) · [Checkpoint revision](https://huggingface.co/vincentleebang/LVR-7B/tree/413c55c908c152f47538320615178cddcf73201a)
