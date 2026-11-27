# Reproducing LVR, LUT and HyLaR

The base checkpoints already have full-suite measurements. Use those predictions and the released LVR-7B evaluation as controls. The full 7B LVR-SFT run and its three decoding budgets are recorded in the [final checkpoint results](../README.md#lvr-7b-after-2500-training-steps). LUT preparation and HyLaR reference checks can proceed independently. [Measured baselines and evaluation protocols](../README.md#benchmark-results)

## LVR

1. Evaluate the step-2,500 SFT checkpoint at 4, 8 and 16 latent steps. Compare each budget with step 500, the released checkpoint and Qwen2.5-VL-7B on identical questions. Report both V* subsets, MMVP question and pair accuracy, and individual BLINK tasks. Inspect paired disagreements and the frequency of complete latent spans.
2. Run the one-update 3B pilot below. Calibrate the frozen-reference objective and checkpoint restart on mixed questions before expanding the reward to the full ViRL answer set. Compare the resulting 1,500-step model with SFT at all three decoding budgets against Table 2. [LVR, §4](https://arxiv.org/html/2509.24251v2#S4)

### Run one 3B latent GRPO update

Use a local LVR 3B SFT checkpoint with its processor and weights. The measured run used the independent step-2,500 checkpoint identified by its two shard hashes in the [3B results](../README.md#independent-3b-checkpoint). Download `lvr_data/virl39k.json` from the LVR annotations and extract the ViRL image archive so that `--images` contains the `ViRL39K/` directory. The command checks the annotation digest and selects the first four single-image rows with A–D answer labels. [LVR annotations](https://drive.google.com/file/d/1RUbKMQU3H7u8iWDqpDv8aiYDMlUaDIlE/view) · [ViRL image revision](https://huggingface.co/datasets/TIGER-Lab/ViRL39K/tree/812ec617dea4bc8a4e751663b88e4ebb7de4d00e)

```sh
uv run --frozen --extra train --extra eval -m methods.lvr.grpo_pilot --checkpoint runs/lvr-3b-sft/model --annotations data/lvr/lvr_data/virl39k.json --images data/lvr/images --output runs/lvr-grpo-pilot.json
```

The pilot samples eight responses per prompt at temperature 0.9, with eight latent steps and a 64-token completion cap. It scores the exact answer letter and the latent-span/answer format independently. The policy forward pass reuses the hidden vectors recorded during generation and scores only discrete actions. At update one the old policy and KL reference both equal the starting SFT model. The single AdamW update uses a constant learning rate of 1e-5, zero weight decay and gradient clipping at 1. Only the language model is updated. The JSON report contains the sampled responses, both rewards and the change in their log probabilities. The updated weights stay in memory for that comparison.


On one A100 80 GB, each of two runs with seed 42 produced the same 32 responses and rewards. All responses used eight latent inputs and earned the format reward. Eight earned the accuracy reward (4, 2, 0 and 2 per question). The unclipped gradient norm was 4.007. Mean absolute change in the sampled discrete actions’ log probabilities after the update was 0.323, averaging first within each response. Before the update, the mean and maximum absolute differences between cached rollout and full-sequence replay were 0.0022 and 0.083 in BF16. The measured update, including image preparation, sampling, replay and backward, took 9.0 seconds and peak allocated GPU memory was 48.5 GiB. These measurements cover four selected prompts and one optimizer update.

### Calibrate multi-step 3B latent GRPO

The calibration uses the same SFT weights, 0.9 temperature and eight generations per question. Each update consumes two single-image A–D questions and two single-image questions with plain numeric answers. The two pools are shuffled independently in sequence with Python seed 42. Numeric scoring compares exact rational values, so `0.5` and `1/2` receive the same accuracy reward. The 64-token completion cap, constant learning rate, strict answer extraction and four prompts per update define this calibration.

The policy samples fresh trajectories at every update. The reference starts from the original SFT checkpoint and stays frozen. Both policies use the recorded latent vectors and the same frozen visual features during replay. The optimizer takes one step per freshly sampled batch. The old policy therefore matches the current policy at the start of each update. The sampled KL estimate is averaged over discrete token actions within each response and then across responses, with coefficient 0.04 in the objective. The calibration also reports the exact forward KL over the vocabulary at these same replayed prefixes.

```sh
uv run --frozen --extra train --extra eval -m methods.lvr.grpo_calibration --checkpoint runs/lvr-3b-sft/model --annotations data/lvr/lvr_data/virl39k.json --images data/lvr/images --output runs/lvr-3b-calibration --steps 6 --save-at 3
```

To repeat updates 4–6 from a saved state, use the same command with `--resume runs/lvr-3b-calibration/checkpoint-3` and a fresh `--output` directory. The checkpoint includes the policy and processor, AdamW state, step, CPU and GPU RNG states, and digests of the SFT input files and selected images. The reference is reloaded from `--checkpoint`. Each invocation writes one record per completed update to `updates.jsonl`. Set `--learning-rate 5e-6` and use a fresh output to run the matched control with the learning rate in the released 3B launcher. The paper specifies 1e-5. [Author launcher](https://github.com/VincentLeebang/lvr/blob/637a9f7e17e865b1372b65f182197fbc4a5a1449/scripts/finetune_lvr_stage2_3b.sh) · [LVR experiment settings](https://arxiv.org/html/2509.24251v2#S4)

Across six updates, each run trained on the same 24 distinct questions and images. The 192 responses comprise 96 letter answers and 96 numeric answers. Accuracy and format below are training rewards. The last column sums forward KL over the entire discrete vocabulary at the final update’s sampled prefixes.

| Learning rate | Correct letters / 96 | Correct numbers / 96 | Full format / 192 | Full format at step 6 / 32 | Sampled KL at step 6 | Conditional KL at step 6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-5 | 43 | 18 | 94 | 0 | 1.7828 | 2.0056 |
| 5e-6 | 44 | 25 | 174 | 28 | 0.0173 | 0.0116 |

After restarting the 1e-5 run at step 3, all 96 subsequent responses and their rewards matched the uninterrupted run. All 824 final policy tensors and 1,302 optimizer tensors matched byte for byte, as did the CPU and GPU RNG states.

On one A100 80 GB, mean update times were 11.9 s at 1e-5 and 12.0 s at 5e-6. At these batch and length settings, 1,500 updates would take approximately 4.9–5.0 GPU hours for rollouts, replay, backward and optimizer steps. Post-update diagnostic replay is timed separately. A checkpoint occupied about 38 GiB and took 95–238 seconds to save. Longer responses and the remaining ViRL answer types require a separate throughput measurement.

## LUT

1. Freeze a seeded 60,000-example Visual-CoT sample with stable IDs and source subsets. Project it to image, question and answer. Record the source revision and sample digest so the vanilla SFT control and LUT teacher see the same examples. Treat the sample as a reproduction choice until the authors' exact IDs can be verified.
2. Start with an eight-latent Qwen2.5-VL-7B teacher. On a small batch, verify that question and answer tokens cannot attend to image tokens, latent tokens can read the image, and only answer positions contribute to token cross-entropy. Overfit the batch and inspect latent decorrelation before scaling.
3. Score the teacher trajectory and a per-step norm-matched random trajectory with the frozen base student. Replay identical answer tokens under teacher forcing. Build the curriculum below. Track retention per subset and compare total retention with the published 93.5%.
4. Distill into a fresh base student with all-layer cosine alignment and answer cross-entropy. Compare with vanilla SFT and shuffled-order distillation. Then add LAPO on the 15K ZwZ-RL-VQA and 9K ViRL mixture. Verify the attribution gates and the sign of latent advantages on a small rollout.
5. Report all four inference budgets (8, 10, 12, 16) and the paper's best-per-benchmark aggregate. The four-suite mean in Table 1 is 68.45 for LUT-SFT and 69.96 after LAPO. Keep MMStar, BLINK and VisualPuzzles as secondary evaluations. [LUT, §§3–4](https://arxiv.org/html/2608.00743v1#S3)

### Prepare the Visual-CoT sample

Point `--images` at the extracted `cot_image_data` directory from the pinned Visual-CoT release. An existing copy with the same subset directories also works. On a fresh machine, download the release's `cot_images_tar_split/cot_images_*` parts and extract their concatenated tar archive in filename order. [Pinned data and image archive](https://huggingface.co/datasets/deepcs233/Visual-CoT/tree/223d2d8c1146fda2bb918801b8276c587b78b61c)

```sh
uv run --frozen -m methods.lut.prepare --images data/cot_image_data --output data/lut/train.jsonl
```

The command downloads and verifies `viscot_363k.json`. Pass `--annotations PATH` to use a local copy. The file contains 404,120 training dialogues. Each output row points to the original image and carries the question and final answer, the subset, the source question ID and a unique ID derived from the source row index. The conversion strips the grounding request and skips the intermediate box and cropped view.

Our default draw samples 60,000 rows without replacement with seed 42. It covers ten subsets and 50,082 distinct image paths. The output SHA-256 is `cfb3eff3d84f80b1dbb089eeaec6e087e983338cbea71e3cf17f02b4bb0e90a9`. The command checks each path and prints the counts and checksum. Use this same JSONL for the LUT teacher and the vanilla SFT control. The LUT paper describes a random draw and does not specify its IDs. [LUT training data](https://arxiv.org/html/2608.00743v1#S4.SS1)

### Build the LUT curriculum

The implemented CPU stage takes frozen-judge scores. Each JSONL row carries a globally unique sample ID, a source subset and two lists of natural log probabilities for the same ordered answer tokens. The scorer must exclude prompt tokens, padding and latent positions from both lists.

```json
{"id":"ocr-17","subset":"ocr","teacher_logprobs":[-0.4,-0.2],"control_logprobs":[-1.2,-0.6]}
```

```sh
uv run --frozen -m methods.lut.curriculum scores.jsonl --seed 42 > curriculum.json
```

The manifest contains input and filtered counts and two ordered ID lists. Consume `low` followed by `high`. Each retained example has mean teacher minus mean control log probability strictly above −0.5. Within each subset, the lower half by rank goes to the first stage. We break ties by sample ID and put the middle example of odd-sized subsets in the second stage. The seed controls shuffling within each stage. Save the score file with the teacher, judge and random-control revisions to reproduce the ordering. [LUT trajectory selection and distillation](https://arxiv.org/html/2608.00743v1#S3.SS2)

## HyLaR

1. Pin the author implementation and verify which processed Zebra-CoT trajectories and weights are available. Audit category counts against the 96,193 SFT examples in Table S2. Inspect a multi-canvas example from source images through token positions and loss masks. Record any differences in filtering or canvas length.
2. Reproduce cold-start SFT with the frozen SigLIP2 encoder and trainable canvas compressor. Verify the reference forward pass on a fixed example and check text/canvas transitions under greedy inference. Compare the 7B checkpoint with the published SFT row before adding RL.
3. Implement DePO on the deduplicated DeepEyes, Thyme and CodeDance mixture. Compare discrete and continuous clipping, the vMF term and the released magnitude convention against the reference on a fixed rollout. Record judge version and rewards. Compare SFT and DePO on the pinned HyLaR suite, then repeat at 3B for the scale comparison. [HyLaR, §§3–4 and appendix](https://arxiv.org/html/2604.20328v2) · [Author implementation](https://github.com/EthenCheng/HyLaR)

## Completing each reproduction

Land each method's pilot, full training recipe and results in reviewable increments. Give each finished method one command from pinned inputs through evaluation. Save checkpoint and data revisions, seed, effective batch, decode settings, per-question predictions and metric denominators. With SGLang and the pinned lmms-eval harness, compare concurrency 16, 32 and 64 on the same high-resolution inputs and record questions per second, GPU utilization and peak memory. Use the fastest stable setting per suite.

## LUT teacher pilot

Run one process per GPU with torchrun. The pilot repeatedly presents the first `WORLD_SIZE` rows of the seeded sample, one per rank, and saves the final teacher and processor. A single-node example follows.

```sh
uv run --frozen --extra train torchrun --standalone --nproc-per-node=8 --module methods.lut.teacher --data data/lut/train.jsonl --images data/cot_image_data --output runs/lut-teacher-pilot
```

Each rank decodes eight continuous steps using the previous final-layer state as the next input. The causal mask restricts access to image keys to image queries and the latent span. The closing boundary predicts the first answer token. Loss combines mean answer cross-entropy and the mean squared off-diagonal cosine similarity of the latent states. Pilot choices include a frozen vision encoder, a 512-token image cap, decorrelation weight 0.1, BF16 weights and 24 AdamW updates at 1e-5. The log averages both terms across ranks. [LUT teacher objective](https://arxiv.org/html/2608.00743v1#S3.SS2)

On 16 A100 80 GB GPUs, the pilot completed 24 updates on the first 16 records of the default 60K sample. Mean answer cross-entropy across ranks was 15.77 on the first update and 10.27 on the last. Averaging the first and last four updates gives 14.39 and 10.51. The decorrelation term moved from 0.333 to 0.281. These measurements describe the training batch. Held-out accuracy and trajectory utility remain to be measured.

### Train and validate the teacher on the full sample

Pass `--full-epoch` to train once through the seeded JSONL in file order. At 16 ranks this gives 3,750 updates on 60,000 examples. Each rank consumes a different example at each update. Use the same torchrun rendezvous settings for training and validation.

```sh
uv run --frozen --extra train torchrun --standalone --nproc-per-node=8 --module methods.lut.teacher --full-epoch --data data/lut/train.jsonl --images data/cot_image_data --output runs/lut-teacher
uv run --frozen --extra train torchrun --standalone --nproc-per-node=8 --module methods.lut.validate_teacher --teacher runs/lut-teacher --train data/lut/train.jsonl --annotations data/viscot_363k.json --images data/cot_image_data
```

Intermediate checkpoints at multiples of 1,000 updates contain the processor, weights and each rank's optimizer state. Resume into a fresh output directory with `--full-epoch --resume-from PATH`, the original data and the same number of ranks.

Validation measures the teacher-specific quantities used for distillation. It selects 512 distinct Visual-CoT images absent from the training sample. The teacher predicts the reference answer through its trained visual bottleneck. A frozen base judge scores those answer tokens with either the teacher trajectory or random vectors matched to its per-step norms. The output `held-out.json` records answer cross-entropy, token accuracy, exact reference-prefix rate, utility differences and retention at the paper's threshold. Prefix matching ends at the last reference token. The report includes per-question values for paired analysis.

The paired trajectory audit compares the two latent index conventions and tests the learned delimiter inputs on disjoint held-out images. [Audit and reproduction commands](lut-trajectory-audit.md)
