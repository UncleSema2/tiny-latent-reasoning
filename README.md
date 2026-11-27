# Tiny Latent Reasoning

Reproductions of latent visual reasoning methods. The first milestone is checkpoint evaluation with SGLang and a pinned lmms-eval submodule.

| Paper | Baseline sizes | Evaluation coverage |
| --- | --- | --- |
| [Latent Visual Reasoning](https://arxiv.org/html/2509.24251v2) | 3B, 7B | V* overall and both subsets, MMVP, and the BLINK subsets used in Tables 1 and 2, including IQ-Test |
| [HyLaR](https://arxiv.org/html/2604.20328v2) | 3B, 7B | V*, HRBench 4K and 8K, MMVP, MMStar, SEED-Bench-2-Plus, BLINK, HallusionBench |
| [LUT](https://arxiv.org/html/2608.00743v1) | 7B | V*, HRBench 4K and 8K, MME-RealWorld-Lite, MMStar, BLINK, VisualPuzzles |

## Reproduction plan

See the [experiment sequence and acceptance checks](docs/reproduction-plan.md) for the next LVR, LUT and HyLaR runs.

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

### LVR-7B results

The local columns use the same pinned suite and processor limits as the 7B base model. The paper columns use the corresponding fixed-step budgets. The local MMVP question and pair metrics use the four corrected labels documented above. [LVR, Table 1](https://arxiv.org/html/2509.24251v2#S4.T1)

| Metric (%) | Local base | Local 4 | Paper 4 | Local 8 | Paper 8 | Local 16 | Paper 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V* overall | 81.68 | 79.06 | 81.2 | 81.15 | 81.7 | 79.58 | 80.6 |
| V* Direct Attributes | 84.35 | 83.48 | 84.4 | 83.48 | 84.4 | 82.61 | 81.7 |
| V* Relative Position | 77.63 | 71.05 | 76.3 | 76.32 | 77.6 | 73.68 | 79.0 |
| MMVP questions | 75.00 | 74.00 | 72.0 | 73.33 | 71.7 | 73.67 | 71.7 |
| MMVP pairs | 51.33 | 48.67 | — | 48.00 | — | 49.33 | — |
| BLINK Counting | 68.33 | 66.67 | 69.2 | 66.67 | 70.0 | 68.33 | 70.8 |
| BLINK IQ-Test | 26.67 | 26.67 | 28.7 | 27.33 | 29.3 | 26.67 | 27.3 |
| BLINK Jigsaw | 68.00 | 52.00 | 52.7 | 56.00 | 52.0 | 58.67 | 52.7 |
| BLINK Relative Reflectance | 34.33 | 29.85 | 42.5 | 31.34 | 42.5 | 34.33 | 41.8 |
| BLINK Spatial Relation | 89.51 | 88.11 | 89.5 | 88.11 | 86.0 | 90.91 | 87.4 |

### LVR-7B throughput

One A100 80 GB, eight latent steps, the first 64 V* high-resolution questions in original order. Each sweep uses the same warmed engine, identical prompt and image paths, and a fixed memory fraction of 0.55. GPU measurements cover batched generation, including image preprocessing. Peak memory is device memory reported by the driver.

| Concurrent requests | Questions/s | Mean GPU utilization (%) | Peak GPU memory (GiB) |
| ---: | ---: | ---: | ---: |
| 16 | 1.46 | 92.9 | 46.86 |
| 32 | 1.47 | 96.6 | 47.10 |
| 64 | 1.50 | 94.8 | 47.12 |

On the first two V* questions, single-request and batched generation both entered one latent span, emitted eight latent placeholders and returned the reference choices A and C. During latent steps the engine masks end-of-sequence tokens to keep the request alive. The reference can emit such a token inside the span. Across all 64 questions, the final answer matched the saved reference on 57, 58 and 58 questions at concurrency 16, 32 and 64, respectively. This comparison measures end-to-end agreement across inference stacks.

## Training LVR with Visual CoT

This command trains the 3B model and evaluates the resulting checkpoint at 4, 8 and 16 latent steps.

```sh
uv run --frozen --extra train --extra eval scripts/lvr_sft.py --size 3b --gpus 8 --data data/lvr --output runs/lvr-3b-sft
```

Use `--size 7b` for the 7B recipe and `--answer-only` for the ordinary SFT control on the same examples. Multiple GPUs on one node use FSDP2. Each rank processes eight adaptive microbatches per optimizer update. Short examples share a padded batch of up to four examples. Sequences of at least 4,096 tokens run individually. The recipe uses 2,500 updates, a frozen vision tower and merger, a peak learning rate of 1e-5, and a cosine schedule. The LVR objective adds 0.1 times the mean squared error on the selected visual embeddings to answer and boundary-token cross-entropy.

The first run fetches the authors' two annotation files and the pinned Visual CoT image archives. Set `HF_HOME` on storage with room for approximately 130 GiB of archive parts plus the extracted images, model weights and optimizer checkpoints. Existing annotations in `lvr_data/` under the data directory are checked by SHA-256. Existing images are read from `images/viscot/`. The release contains 417,453 single-image training records. [Author annotations](https://github.com/VincentLeebang/lvr/blob/637a9f7e17e865b1372b65f182197fbc4a5a1449/README.md#dataset-preparation) · [Visual CoT image revision](https://huggingface.co/datasets/deepcs233/Visual-CoT/tree/223d2d8c1146fda2bb918801b8276c587b78b61c)

The one-GPU pilot trains on a fixed four-example subset.

```sh
uv run --frozen --extra train -m methods.lvr.training --data data/lvr --output runs/lvr-3b-overfit --overfit 4 --batch-size 4 --accumulation 1 --steps 24
```

All 824 tensors in the 3B base checkpoint and all 729 tensors in the 7B base checkpoint matched their source values before training. On one A100 80 GB, the 3B pilot took 47.6 seconds for 24 updates. Text cross-entropy fell from 5.635 to 1.372 and visual reconstruction MSE from 23.736 to 7.862 on the four training examples. The collator matched the Qwen processor prompt tokens and image grids for all four.

Decode the same four questions with SGLang to inspect the generated text, count latent start markers and check the answer suffix against the annotation.

```sh
uv run --frozen --extra train --extra eval -m methods.lvr.training.check_overfit --data data/lvr --checkpoint runs/lvr-3b-overfit
```

With eight latent steps allowed, all four generated answers matched the targets after 24 updates. The LVR pilot started each response with `<|im_start|>` and generated zero latent spans. The matched answer-only control also produced all four target answers.

Repeating the training command from the base checkpoint with `--steps 96` and a fresh output directory reduced text cross-entropy to 0.453 and region MSE to 5.522. All four responses began with `<|lvr_start|>`. Zero of four ended with the target answer at the eight-step decoding budget. The teacher-forced region lengths on these four records were 52, 12, 4 and 112 tokens.

### LVR-7B after 500 training steps

This command runs the first 500 updates of the 2,500-step schedule and evaluates the saved weights at all three decoding budgets.

```sh
uv run --frozen --extra train --extra eval scripts/lvr_sft.py --size 7b --gpus 8 --data data/lvr --output runs/lvr-7b-sft --steps 2500 --stop-after 500
```

The checkpoint includes the optimizer, scheduler and RNG state for each rank. To train the remaining 2,000 updates, repeat the command with `--resume` in place of `--stop-after 500`. For a distributed FSDP restart, the training module accepts a different world size if the number of packs per optimizer update stays fixed. For example, 16 ranks with four accumulation steps preserve the batch of eight ranks with eight accumulation steps. The training manifest records the change. Floating-point reductions can differ across layouts. The run to step 500 took 4 h 10 min on eight A100 80 GB GPUs, including startup and saving. The saved model has SHA-256 prefix `bcd5762f081af251`.

Each decoding budget produced 1,379 predictions on the pinned LVR suite. The V* subset tasks repeat questions from the overall split. The released column uses the authors’ checkpoint on the same local harness. All columns report percentages.

| Metric | Base | Released 8 | Step 500, 4 | Step 500, 8 | Step 500, 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V* overall | 81.68 | 81.15 | 82.20 | 81.15 | 81.68 |
| V* Direct Attributes | 84.35 | 83.48 | 86.09 | 83.48 | 83.48 |
| V* Relative Position | 77.63 | 76.32 | 76.32 | 77.63 | 76.32 |
| MMVP questions | 75.00 | 73.33 | 71.33 | 71.67 | 74.67 |
| MMVP pairs | 51.33 | 48.00 | 44.67 | 45.33 | 50.67 |
| BLINK Counting | 68.33 | 66.67 | 68.33 | 70.00 | 71.67 |
| BLINK IQ-Test | 26.67 | 27.33 | 23.33 | 24.00 | 23.33 |
| BLINK Jigsaw | 68.00 | 56.00 | 72.00 | 73.33 | 75.33 |
| BLINK Relative Reflectance | 34.33 | 31.34 | 26.12 | 26.87 | 26.87 |
| BLINK Spatial Relation | 89.51 | 88.11 | 86.01 | 85.31 | 85.31 |

At eight latent steps, the step-500 model and the released model both answered 155 of 191 V* questions correctly. Each solved ten questions that the other missed. On Jigsaw the step-500 model solved 34 questions that the released model missed and missed eight that it solved. On Relative Reflectance the corresponding counts were four and ten. In a separate greedy decoding check on all 191 V* images, every step-500 response opened and closed a latent span and returned a letter inside the answer tags.

On the first 64 high-resolution V* questions, the step-500 model achieved the following throughput at eight latent steps on one A100 80 GB. We used concurrency 64 for the suite.

| Concurrent requests | Questions/s | Mean GPU utilization (%) | Peak GPU memory (GiB) |
| ---: | ---: | ---: | ---: |
| 16 | 1.479 | 95.8 | 46.80 |
| 32 | 1.488 | 92.3 | 46.87 |
| 64 | 1.510 | 97.1 | 46.89 |

### LVR-7B after 2,500 training steps

We resumed the step-500 checkpoint on 16 A100 80 GB GPUs with four accumulation steps per rank, keeping 64 adaptive packs per optimizer update. The remaining 2,000 updates and final save took about ten hours. Each evaluation produced 1,379 predictions at its fixed decoding budget on the same questions and targets as the base and released checkpoints. V* subset questions also appear in the overall split. Scores are percentages.

| Metric | Base | Released, 8 | Step 500, 8 | Final, 4 | Final, 8 | Final, 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V* overall | 81.68 | 81.15 | 81.15 | 82.20 | 82.20 | 81.15 |
| V* Direct Attributes | 84.35 | 83.48 | 83.48 | 83.48 | 83.48 | 84.35 |
| V* Relative Position | 77.63 | 76.32 | 77.63 | 80.26 | 80.26 | 78.95 |
| MMVP questions | 75.00 | 73.33 | 71.67 | 74.33 | 74.00 | 74.67 |
| MMVP pairs | 51.33 | 48.00 | 45.33 | 50.67 | 50.00 | 50.67 |
| BLINK Counting | 68.33 | 66.67 | 70.00 | 69.17 | 70.00 | 68.33 |
| BLINK IQ-Test | 26.67 | 27.33 | 24.00 | 24.00 | 25.33 | 24.00 |
| BLINK Jigsaw | 68.00 | 56.00 | 73.33 | 57.33 | 60.67 | 64.67 |
| BLINK Relative Reflectance | 34.33 | 31.34 | 26.87 | 29.85 | 32.09 | 30.60 |
| BLINK Spatial Relation | 89.51 | 88.11 | 85.31 | 88.11 | 87.41 | 88.81 |

At eight latent steps, the final model answered 157/191 V* questions correctly, gaining seven and losing five relative to step 500. On BLINK Jigsaw it fell from 110/150 to 91/150, gaining eight and losing 27. We matched the source question, target and document hash, and independently read the answer letter in all 150 final responses at each budget. All 450 letters agreed with the evaluator. Of the 27 losses at eight steps, 23 responses contained only a letter inside the answer tags and four added a matching image description. The final model chose the second image (A) on 23 of the 27 losses.

### Independent 3B checkpoint

We also evaluated a previously trained 3B checkpoint from a local adaptation of the authors' implementation, with seed 42, 2,500 optimizer updates and 64 adaptive microbatches per update. The two local weight shards have SHA-256 prefixes `79ad32ff7de9dc8f` and `4888ff83d5aebe29`, in shard order. The 3B scores below describe this independent checkpoint. [Reference implementation](https://github.com/VincentLeebang/lvr/tree/637a9f7e17e865b1372b65f182197fbc4a5a1449) · [LVR, Table 2](https://arxiv.org/html/2509.24251v2#S4.T2)

Each local column uses the pinned 3B suite and the MMVP corrections described above. This command reproduces the eight-step column with the checkpoint in `$CHECKPOINT`.

```sh
uv run --frozen --extra eval scripts/baseline.py lvr --size 3b --checkpoint "$CHECKPOINT" --lvr-steps 8 --batch-size 64 --output results/lvr-3b-8
```

| Metric (%) | Local base | Local 4 | Paper 4 | Local 8 | Paper 8 | Local 16 | Paper 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V* overall | 75.39 | 68.59 | 64.9 | 69.11 | 65.5 | 69.63 | 66.5 |
| V* Direct Attributes | 84.35 | 74.78 | 69.6 | 74.78 | 71.3 | 76.52 | 71.3 |
| V* Relative Position | 61.84 | 60.53 | 60.5 | 59.21 | 60.5 | 59.21 | 56.6 |
| MMVP question accuracy | 64.00 | 64.67 | 54.7 | 65.33 | 56.0 | 65.00 | 56.0 |
| MMVP pair accuracy | 34.00 | 33.33 | — | 34.67 | — | 36.00 | — |
| BLINK IQ-Test | 26.67 | 30.00 | 29.3 | 31.33 | 30.7 | 29.33 | 30.0 |
| BLINK Jigsaw | 38.67 | 52.67 | 52.7 | 52.67 | 52.7 | 52.67 | 52.0 |

For this checkpoint, eight-step decoding on the same 64 high-resolution V* questions gave the following throughput on one A100 80 GB. We used 64 concurrent requests in the full evaluation.

| Concurrent requests | Questions/s | Mean GPU utilization (%) | Peak GPU memory (GiB) |
| ---: | ---: | ---: | ---: |
| 16 | 1.715 | 86.4 | 42.75 |
| 32 | 1.894 | 93.0 | 42.78 |
| 64 | 1.913 | 95.3 | 42.78 |
