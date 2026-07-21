"""
Build a POPE-like training dataset from COCO 2014 train set.

Produces 3000 questions (500 images × 6 questions each: 3 yes + 3 no)
across three negative-sampling strategies: random, popular, adversarial.

Output format matches the standard POPE eval format exactly:
  JSONL with {"question_id": int, "image": str, "text": str, "label": "yes"|"no"}

Usage:
    python scripts/build_pope_train.py \
        --coco-json data/coco2014/captions/annotations/instances_train2014.json \
        --num-images 500 \
        --num-samples 3 \
        --seed 42 \
        --output-dir pope_train
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build POPE-like training data from COCO")
    parser.add_argument("--coco-json", required=True,
                        help="Path to COCO instances JSON (e.g. instances_train2014.json)")
    parser.add_argument("--num-images", type=int, default=500,
                        help="Number of images to sample (default: 500 → 3000 questions)")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of positive (and negative) questions per image (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="pope_train",
                        help="Directory to write output JSONL files")
    parser.add_argument("--img-root", default=None,
                        help="Optional root path to prepend to image filenames")
    return parser.parse_args()


def load_coco_objects(coco_json_path):
    """
    Parse COCO instances JSON and return:
      img_objects: dict[image_id] -> set of category names present
      img_id_to_file: dict[image_id] -> file_name
      all_objects: set of all category names in the dataset
    """
    with open(coco_json_path) as f:
        data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in data["categories"]}

    img_objects = defaultdict(set)
    for ann in data["annotations"]:
        cat_name = cat_map[ann["category_id"]]
        img_objects[ann["image_id"]].add(cat_name)

    img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    all_objects = set(cat_map.values())

    return img_objects, img_id_to_file, all_objects


def build_statistics(img_objects):
    """
    Build frequency dict and co-occurrence matrix across all images.
      obj_freq:  {object_name: count}  — how many images contain this object
      co_occur:  {obj_a: {obj_b: count}}  — how many images contain both obj_a and obj_b
    """
    obj_freq = Counter()
    co_occur = defaultdict(Counter)

    for objects in img_objects.values():
        obj_list = list(objects)
        for obj in obj_list:
            obj_freq[obj] += 1
        # Count all unordered pairs
        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                co_occur[obj_list[i]][obj_list[j]] += 1
                co_occur[obj_list[j]][obj_list[i]] += 1

    return obj_freq, co_occur


def sample_negative(neg_strategy, pos_object, image_objects, history_objects,
                    gt_objects_list, sorted_objects, sorted_co_occur):
    """
    Sample a single negative object (NOT in the image and NOT in history)
    according to the chosen strategy.

    Returns the sampled object name, or None if sampling fails.
    """
    forbidden = image_objects | set(history_objects)

    if neg_strategy == "random":
        # Uniform random from all objects, excluding forbidden ones
        candidates = [o for o in gt_objects_list if o not in forbidden]
        if candidates:
            return random.choice(candidates)

    elif neg_strategy == "popular":
        # Pick the most frequent objects first (descending frequency)
        for obj, _ in sorted_objects:
            if obj not in forbidden:
                return obj
        # Fallback: random from remaining
        candidates = [o for o in gt_objects_list if o not in forbidden]
        if candidates:
            return random.choice(candidates)

    elif neg_strategy == "adversarial":
        # From co-occurring objects with pos_object, pick most frequent co-occurrer
        if pos_object in sorted_co_occur:
            for obj, _ in sorted_co_occur[pos_object]:
                if obj not in forbidden:
                    return obj
        # Fallback: random
        candidates = [o for o in gt_objects_list if o not in forbidden]
        if candidates:
            return random.choice(candidates)

    return None


def create_question(qid, image_name, object_name, label, template="Is there a {} in the image?"):
    """Create a single POPE-format question dict."""
    # Handle a/an
    if object_name[0].lower() in "aeiou":
        text = template.replace(" a ", " an ").format(object_name)
    else:
        text = template.format(object_name)
    return {
        "question_id": qid,
        "image": image_name,
        "text": text,
        "label": label,
    }


def generate_pope(img_objects, img_id_to_file, sampled_image_ids,
                  obj_freq, co_occur, all_objects, num_samples, neg_strategy):
    """
    Generate POPE questions for a set of sampled images using the given
    negative-sampling strategy.

    Returns a list of question dicts.
    """
    # Pre-sort objects for efficient lookup
    gt_objects_list = list(all_objects)
    sorted_objects = sorted(obj_freq.items(), key=lambda x: x[1], reverse=True)
    sorted_co_occur = {
        obj: sorted(co.items(), key=lambda x: x[1], reverse=True)
        for obj, co in co_occur.items()
    }

    questions = []
    qid = 0

    for img_id in sampled_image_ids:
        image_objects = list(img_objects[img_id])
        image_name = img_id_to_file[img_id]

        # Need at least num_samples objects in the image to create positive questions
        if len(image_objects) < num_samples:
            continue

        # Randomly pick num_samples positive objects (without replacement)
        pos_objects = random.sample(image_objects, num_samples)

        history_objects = []  # track objects used for THIS image to avoid repeats

        for pos_obj in pos_objects:
            qid += 1
            # Positive question
            questions.append(create_question(qid, image_name, pos_obj, "yes"))
            history_objects.append(pos_obj)

            # Negative question — sample an object NOT present in the image
            neg_obj = sample_negative(
                neg_strategy, pos_obj, set(image_objects), history_objects,
                gt_objects_list, sorted_objects, sorted_co_occur
            )
            if neg_obj is None:
                # Fallback: pick any object not in the image
                for obj in gt_objects_list:
                    if obj not in set(image_objects) | set(history_objects):
                        neg_obj = obj
                        break
            if neg_obj is None:
                # Last resort: just pick something
                neg_obj = random.choice(gt_objects_list)

            qid += 1
            questions.append(create_question(qid, image_name, neg_obj, "no"))
            history_objects.append(neg_obj)

    return questions


def main():
    args = parse_args()
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Load COCO annotations and extract per-image object sets
    # ------------------------------------------------------------------
    print(f"Loading COCO annotations from {args.coco_json} ...")
    img_objects, img_id_to_file, all_objects = load_coco_objects(args.coco_json)
    print(f"  {len(img_objects)} images with annotations")
    print(f"  {len(all_objects)} unique object categories")

    # ------------------------------------------------------------------
    # 2. Filter images that have enough objects
    # ------------------------------------------------------------------
    eligible = [
        img_id for img_id, objs in img_objects.items()
        if len(objs) >= args.num_samples
    ]
    print(f"  {len(eligible)} images with >= {args.num_samples} objects")

    if len(eligible) < args.num_images:
        print(f"WARNING: only {len(eligible)} eligible images, but {args.num_images} requested.")
        args.num_images = len(eligible)

    # ------------------------------------------------------------------
    # 3. Sample images (fixed seed for reproducibility)
    # ------------------------------------------------------------------
    sampled = random.sample(eligible, args.num_images)
    print(f"  Sampled {len(sampled)} images")

    # ------------------------------------------------------------------
    # 4. Build frequency + co-occurrence statistics
    # ------------------------------------------------------------------
    print("Building object statistics (frequency + co-occurrence) ...")
    obj_freq, co_occur = build_statistics(img_objects)
    print(f"  Top-10 objects: {obj_freq.most_common(10)}")

    # ------------------------------------------------------------------
    # 5. Generate POPE questions for each strategy
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    for strategy in ["random", "popular", "adversarial"]:
        print(f"Generating POPE ({strategy}) questions ...")
        questions = generate_pope(
            img_objects, img_id_to_file, sampled,
            obj_freq, co_occur, all_objects,
            args.num_samples, strategy
        )

        out_path = os.path.join(args.output_dir, f"coco_train_pope_{strategy}.json")
        with open(out_path, "w") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")

        yes_count = sum(1 for q in questions if q["label"] == "yes")
        no_count = sum(1 for q in questions if q["label"] == "no")
        unique_imgs = len(set(q["image"] for q in questions))
        print(f"  → {out_path}")
        print(f"    {len(questions)} questions, {unique_imgs} images, "
              f"{yes_count} yes, {no_count} no")

    # ------------------------------------------------------------------
    # 6. Save auxiliary statistics files (so you can inspect / reuse)
    # ------------------------------------------------------------------
    stats = {
        "num_images_sampled": args.num_images,
        "num_samples_per_image": args.num_samples,
        "total_questions_per_strategy": args.num_images * args.num_samples * 2,
        "sampled_image_ids": sampled,
        "object_frequencies": dict(obj_freq.most_common()),
        "seed": args.seed,
    }
    stats_path = os.path.join(args.output_dir, "pope_train_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
