# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research codebase for studying **VLM causality** — whether vision-language models truly understand object presence in images or rely on spurious correlations (co-occurrence statistics). Uses the **POPE** (Polling-based Object Probing Evaluation) protocol with COCO 2014 images.

The core experiment: fine-tune LLaVA on POPE-style yes/no questions from the COCO **train** set, then test generalization on the standard POPE **eval** set to measure how much the model learns causal object recognition vs. dataset biases.

## Architecture

```
scripts/
  build_pope_train.py     # Generate POPE training data from COCO train set
  llava_infer_baseline.py # Run LLaVA 1.5 7B batched inference on POPE data
  eval_pope.py            # Compute accuracy/precision/recall/F1/yes-ratio
  download_coco2014.py    # Download COCO 2014 via kagglehub
pope/                     # Standard POPE eval sets (500 images × 6 questions each)
pope_train/               # Generated POPE training sets (from COCO train2014)
results/                  # Model inference outputs (JSONL)
```

**Data flow:** COCO annotations → `build_pope_train.py` → `pope_train/*.json` → `llava_infer_baseline.py` → `results/*.json` → `eval_pope.py` → metrics

**Three negative-sampling strategies** (set at build time, used in both train and eval):
- `random` — uniform random negative object
- `popular` — most frequent COCO objects as negatives
- `adversarial` — objects that most frequently co-occur with the positive object

## Key dependencies

- **PyTorch 2.11 + CUDA 13** — GPU inference/training
- **Hugging Face Transformers** — LLaVA model loading (`LlavaForConditionalGeneration`)
- **nnsight** (`0.6.3`) — neuron/activation analysis for causal intervention experiments
- **pycocotools, kagglehub** — dataset access
- **bitsandbytes** — optional quantization

## Common commands

```bash
# Activate Python environment
source /venv/main/bin/activate

# Download COCO 2014 dataset
python scripts/download_coco2014.py

# Build POPE training data (3000 questions = 500 images × 6 Q each)
python scripts/build_pope_train.py \
    --coco-json data/coco2014/captions/annotations/instances_train2014.json \
    --num-images 500 --num-samples 3 --seed 42

# Run LLaVA 1.5 7B batched inference on POPE data
python scripts/llava_infer_baseline.py \
    --pope-file pope_train/coco_train_pope_adversarial.json \
    --image-root data/coco2014/train2014/train2014 \
    --batch-size 4 \
    --output results/llava15_7b_adversarial_train.json

# Evaluate model answers against POPE labels
python scripts/eval_pope.py \
    --ans-file results/llava15_7b_adversarial_train.json \
    --label-file pope_train/coco_train_pope_adversarial.json

# Evaluate with per-question error breakdown
python scripts/eval_pope.py --ans-file ... --label-file ... --verbose
```

## Data formats

**POPE label files** (JSONL): `{"question_id": int, "image": "COCO_...jpg", "text": "Is there a ...?", "label": "yes"|"no"}`

**Model answer files** (JSONL): `{"question_id": int, "question": "...", "answer": "yes"|"no"}`

Answer extraction (`llava_infer_baseline.py`) and evaluation binarization (`eval_pope.py`) each implement independent yes/no heuristics — `llava_infer_baseline.py` uses regex word-boundary matching, while `eval_pope.py` mimics the standard POPE `evaluate.py` logic (first sentence, strip commas, check for "no"/"not"). Both are intentionally simple since the LLaVA prompt instructs "Answer with only Yes or No."
