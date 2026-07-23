"""
Run LLaVA 1.5 7B (HF) inference on a POPE JSONL file.

Saves results incrementally:
  JSONL with {"question_id": int, "question": str, "answer": "yes"|"no", "raw_output": str}

Supports real batched inference for speed.

Usage:
    python scripts/llava_infer_baseline.py \
        --pope-file pope_train/coco_train_pope_adversarial.json \
        --image-root data/coco2014/train2014/train2014 \
        --batch-size 4 \
        --output results/llava15_7b_adversarial_train.json
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA 1.5 7B POPE inference")
    parser.add_argument("--pope-file", required=True,
                        help="Path to POPE JSONL file")
    parser.add_argument("--image-root", required=True,
                        help="Directory containing COCO images")
    parser.add_argument("--output", required=True,
                        help="Path to save results (JSONL)")
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf",
                        help="HF model id")
    parser.add_argument("--max-new-tokens", type=int, default=4,
                        help="Max tokens to generate (answer is just yes/no)")
    parser.add_argument("--save-interval", type=int, default=500,
                        help="Save results every N questions")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of samples per forward pass")
    parser.add_argument("--torch-dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    return parser.parse_args()


def load_pope(filepath):
    """Load POPE JSONL into list of dicts."""
    with open(filepath) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(question_text):
    """Build a LLaVA-format prompt that strongly encourages a yes/no answer."""
    return (
        f"USER: <image>\n"
        f"{question_text}\n"
        f"ASSISTANT:"
    )


def extract_answer(text):
    """Extract yes/no from model output."""
    text = text.strip().lower()

    # Direct match (most common: model outputs just "Yes" or "No")
    if text.startswith("yes") or text.endswith("yes"):
        return "yes"
    if text.startswith("no") or text.endswith("no"):
        return "no"

    # Pattern: "Yes, ..." or "No, ..."
    if re.search(r'\byes\b', text):
        return "yes"
    if re.search(r'\bno\b', text):
        return "no"

    # Fallback
    return text.strip()


def load_model_and_processor(model_id, torch_dtype_str):
    """Load LLaVA model and processor from HF."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[torch_dtype_str]

    print(f"Loading model: {model_id}  (dtype={torch_dtype_str})")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"  Model loaded. Device: {model.device}")
    return model, processor


def run_batched_inference(model, processor, batch_items, image_root, max_new_tokens):
    """
    Run inference on a batch of POPE items.

    Args:
        batch_items: list of dicts with "image" and "text" keys
        image_root: path prefix for images
        max_new_tokens: generation limit

    Returns:
        list of decoded answer strings
    """
    images = []
    prompts = []

    for item in batch_items:
        img_path = os.path.join(image_root, item["image"])
        images.append(Image.open(img_path).convert("RGB"))
        prompts.append(build_prompt(item["text"]))

    # Process as batch: list of texts + list of images, with padding
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode each output (strip the input prompt part)
    answers = []
    raw_outputs = []
    for i in range(len(batch_items)):
        input_len = inputs["input_ids"][i].shape[0]
        generated_ids = outputs[i, input_len:]
        raw_text = processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        raw_outputs.append(raw_text)
        answers.append(extract_answer(raw_text))

    return answers, raw_outputs


def save_results(results, output_path):
    """Write results to JSONL."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load POPE data
    # ------------------------------------------------------------------
    print(f"Loading POPE data from {args.pope_file}")
    pope_data = load_pope(args.pope_file)
    print(f"  {len(pope_data)} questions")

    # ------------------------------------------------------------------
    # 2. Verify images exist
    # ------------------------------------------------------------------
    unique_images = set(q["image"] for q in pope_data)
    print(f"  {len(unique_images)} unique images")

    missing_images = set()
    for img_name in unique_images:
        if not os.path.exists(os.path.join(args.image_root, img_name)):
            missing_images.add(img_name)
    if missing_images:
        print(f"  ERROR: {len(missing_images)} images missing!")
        for m in list(missing_images)[:5]:
            print(f"    {m}")
        sys.exit(1)
    print(f"  All {len(unique_images)} images found on disk")

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    model, processor = load_model_and_processor(args.model_id, args.torch_dtype)

    # ------------------------------------------------------------------
    # 4. Run batched inference
    # ------------------------------------------------------------------
    results = []

    # Use a manual loop with range for batching
    bs = args.batch_size
    total = len(pope_data)

    pbar = tqdm(total=total, desc="Inference")
    i = 0
    while i < total:
        # Collect batch
        batch_end = min(i + bs, total)
        batch_items = pope_data[i:batch_end]

        # Run batched forward pass
        answers, raw_outputs = run_batched_inference(
            model, processor, batch_items,
            args.image_root, args.max_new_tokens,
        )

        # Collect results
        for item, answer, raw_output in zip(batch_items, answers, raw_outputs):
            results.append({
                "question_id": item["question_id"],
                "question": item["text"],
                "answer": answer,
                "raw_output": raw_output,
            })

        # Incremental save
        if len(results) % args.save_interval < bs or batch_end == total:
            save_results(results, args.output)
            tqdm.write(f"  Saved {len(results)} results to {args.output}")

        i = batch_end
        pbar.update(len(batch_items))

    pbar.close()

    # ------------------------------------------------------------------
    # 5. Final save & stats
    # ------------------------------------------------------------------
    save_results(results, args.output)
    print(f"\nFinal: {len(results)} results saved to {args.output}")

    yes_count = sum(1 for r in results if r["answer"] == "yes")
    no_count = sum(1 for r in results if r["answer"] == "no")
    other_count = len(results) - yes_count - no_count
    print(f"  Yes: {yes_count}, No: {no_count}, Other: {other_count}")


if __name__ == "__main__":
    main()
