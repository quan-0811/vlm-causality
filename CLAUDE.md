# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research codebase for studying **VLM causality** — whether vision-language models truly understand object presence in images or rely on spurious correlations (co-occurrence statistics). Uses the **POPE** (Polling-based Object Probing Evaluation) protocol with COCO 2014 images.

The core experiment: fine-tune LLaVA on POPE-style yes/no questions from the COCO **train** set, then test generalization on the standard POPE **eval** set to measure how much the model learns causal object recognition vs. dataset biases.

## Architecture

```
scripts/
  build_pope_train.py        # Generate POPE training data from COCO train set
  llava_infer_baseline.py    # Run LLaVA 1.5 7B batched inference on POPE data
  eval_pope.py               # Compute accuracy/precision/recall/F1/yes-ratio
  download_coco2014.py       # Download COCO 2014 via kagglehub
pope/                        # Standard POPE eval sets (JSON arrays; 6 questions per image)
pope_train/                  # Generated POPE training sets (JSONL) + stats file
results/
  train/                     # Model answers on training questions
  val/                       # Model answers on standard POPE eval questions
data/                        # COCO 2014 images + annotations (gitignored)
```

**Data flow (train):** COCO train2014 annotations → `build_pope_train.py` → `pope_train/*.jsonl` → `llava_infer_baseline.py` → `results/train/*.jsonl` → `eval_pope.py` → metrics

**Data flow (eval):** Standard POPE eval sets in `pope/*.json` → `llava_infer_baseline.py` → `results/val/*.jsonl` → `eval_pope.py` → metrics

**Three negative-sampling strategies** (set at build time):
- `random` — uniform random negative object
- `popular` — most frequent COCO objects as negatives
- `adversarial` — objects that most frequently co-occur with the positive object (hardest)

**Two different yes/no heuristics — important for debugging:**
- `llava_infer_baseline.py::extract_answer()` uses regex word-boundary matching on the full output
- `eval_pope.py::binarize_answer()` mimics the standard POPE `evaluate.py` logic: first sentence only, strip commas, check for "no"/"not" words
- These can produce different results on the same raw text; the inference heuristic is the one that matters for the saved answers

## Key dependencies

- **PyTorch 2.11 + CUDA 13** — GPU inference/training
- **Hugging Face Transformers** — LLaVA model loading (`LlavaForConditionalGeneration`, `AutoProcessor`)
- **nnsight** (0.6.3) — installed for future causal intervention / activation analysis experiments (not yet used in scripts)
- **pycocotools, kagglehub** — dataset access
- **bitsandbytes** — optional quantization

## Common commands

```bash
# Activate Python environment
source /venv/main/bin/activate

# Download COCO 2014 dataset (uses kagglehub, writes to data/coco2014/)
python scripts/download_coco2014.py

# Build POPE training data (each strategy produces a separate file; 3 questions/image → 6 Q total per image)
python scripts/build_pope_train.py \
    --coco-json data/coco2014/captions/annotations/instances_train2014.json \
    --num-images 500 --num-samples 3 --seed 42

# Run LLaVA 1.5 7B batched inference on POPE data
python scripts/llava_infer_baseline.py \
    --pope-file pope_train/coco_train_pope_adversarial.jsonl \
    --image-root data/coco2014/train2014/train2014 \
    --batch-size 4 \
    --output results/train/llava15_7b_adversarial_train.jsonl

# Evaluate model answers against POPE labels
python scripts/eval_pope.py \
    --ans-file results/train/llava15_7b_adversarial_train.jsonl \
    --label-file pope_train/coco_train_pope_adversarial.jsonl

# Evaluate with per-question error breakdown
python scripts/eval_pope.py --ans-file ... --label-file ... --verbose
```

## Data formats

**Standard POPE eval files** (`pope/*.json`): JSON arrays (not JSONL). Each entry: `{"question_id": int, "image": "COCO_val2014_...jpg", "text": "Is there a ...?", "label": "yes"|"no"}`

**POPE train files** (`pope_train/*.jsonl`): JSONL, one JSON object per line. Same schema as eval files but referencing train2014 images.

**Model answer files** (`results/**/*.jsonl`): JSONL. `{"question_id": int, "question": "Is there a ...?", "answer": "yes"|"no"}`

**Stats file** (`pope_train/pope_train_stats.json`): Records sampled image IDs, object frequency distribution, and generation parameters. Useful for verifying reproducibility.

## GPU considerations

This codebase runs on a Vast.ai instance with CUDA 13 on NVIDIA GPUs. The `requirements.txt` pins `torch==2.11.0` with CUDA 13 bindings. When adding packages, ensure CUDA compatibility with the installed driver — use `vast-capabilities | jq '.hardware.gpu.cuda'` to check the environment. The `data/` directory is gitignored; COCO images must be downloaded separately.
