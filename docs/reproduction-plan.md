# Reproducing LVR, LUT and HyLaR

The base checkpoints already have full-suite measurements. Use those predictions and the released LVR-7B evaluation as controls. Finish the running LVR experiment on its training branch. LUT preparation and HyLaR reference checks can proceed independently. [Measured baselines and evaluation protocols](../README.md#benchmark-results)

## LVR

1. Evaluate the step-2,500 SFT checkpoint at 4, 8 and 16 latent steps. Compare each budget with step 500, the released checkpoint and Qwen2.5-VL-7B on identical questions. Report both V* subsets, MMVP question and pair accuracy, and individual BLINK tasks. Inspect paired disagreements and the frequency of complete latent spans.
2. Reproduce the paper's RL experiment on **3B**. Start from an identified SFT checkpoint, pin the ViRL data and check the format and accuracy rewards on sampled responses. Run a small rollout and optimizer update before the 1,500-step run. Compare SFT and latent GRPO at each of the three budgets against Table 2. [LVR, §4](https://arxiv.org/html/2509.24251v2#S4)

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
