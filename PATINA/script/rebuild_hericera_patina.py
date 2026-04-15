#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


DEFAULT_SEED = 20260319


@dataclass
class SampleRecord:
    sample_id: str
    class_name: str
    source_dataset: str
    source_code: str
    source_object_id: str
    image_path: Path
    record_path: Path
    image_url: str | None
    object_url: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a PATINA training pack from the HeriCera-6C dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("../HeriCera-6C_raw data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../datasets/hericera6c"),
    )
    parser.add_argument("--max-square-side", type=int, default=1024)
    parser.add_argument("--context-pad-ratio", type=float, default=0.08)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--train-mask-count", type=int, default=12000)
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1))))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def collect_border_strip(array: np.ndarray, strip: int) -> np.ndarray:
    top = array[:strip, :, :].reshape(-1, 3)
    bottom = array[-strip:, :, :].reshape(-1, 3)
    left = array[:, :strip, :].reshape(-1, 3)
    right = array[:, -strip:, :].reshape(-1, 3)
    return np.concatenate([top, bottom, left, right], axis=0)


def collect_border_diff(diff_map: np.ndarray, strip: int) -> np.ndarray:
    top = diff_map[:strip, :].reshape(-1)
    bottom = diff_map[-strip:, :].reshape(-1)
    left = diff_map[:, :strip].reshape(-1)
    right = diff_map[:, -strip:].reshape(-1)
    return np.concatenate([top, bottom, left, right], axis=0)


def median_border_color(array: np.ndarray) -> tuple[int, int, int]:
    strip = max(2, min(array.shape[0], array.shape[1]) // 30)
    border = collect_border_strip(array, strip)
    return tuple(int(v) for v in np.median(border, axis=0).round())


def fill_holes(mask: np.ndarray) -> np.ndarray:
    filled = mask.copy()
    h, w = filled.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood, (0, 0), 255)
    holes = cv2.bitwise_not(filled)
    return mask | holes


def choose_best_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    h, w = mask.shape
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    best_label = 0
    best_score = -1.0
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        centroid = centroids[label].astype(np.float32)
        distance = np.linalg.norm(centroid - center) / max(np.linalg.norm(center), 1.0)
        score = area * (1.25 - min(distance, 1.0))
        if score > best_score:
            best_score = score
            best_label = label

    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def grabcut_foreground(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    rect_margin = max(4, int(round(min(h, w) * 0.04)))
    rect = (
        rect_margin,
        rect_margin,
        max(1, w - rect_margin * 2),
        max(1, h - rect_margin * 2),
    )
    gc_mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, gc_mask, rect, bg_model, fg_model, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.full((h, w), 255, np.uint8)

    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    if mask.max() == 0:
        return np.full((h, w), 255, np.uint8)
    return mask


def detect_foreground_mask(rgb: np.ndarray, use_grabcut_fallback: bool = False) -> tuple[np.ndarray, tuple[int, int, int], str]:
    h, w = rgb.shape[:2]
    strip = max(2, int(round(min(h, w) * 0.03)))
    bg_rgb = np.array(median_border_color(rgb), dtype=np.uint8)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    diff = np.linalg.norm(lab - bg_lab, axis=2)
    border_diff = collect_border_diff(diff, strip)
    diff_threshold = max(10.0, float(np.percentile(border_diff, 90)) + 8.0)
    diff_mask = diff > diff_threshold

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 40, 120)
    edge_kernel = np.ones((ensure_odd(min(h, w) * 0.012), ensure_odd(min(h, w) * 0.012)), np.uint8)
    canny = cv2.dilate(canny, edge_kernel, iterations=1)

    merged = np.where(diff_mask | (canny > 0), 255, 0).astype(np.uint8)
    close_kernel = np.ones((ensure_odd(min(h, w) * 0.02), ensure_odd(min(h, w) * 0.02)), np.uint8)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, close_kernel)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    merged = fill_holes(merged)
    merged = choose_best_component(merged)

    area_ratio = float(np.count_nonzero(merged)) / float(h * w)
    method = "border_diff"
    if use_grabcut_fallback and (area_ratio < 0.02 or area_ratio > 0.95):
        merged = grabcut_foreground(rgb)
        merged = cv2.morphologyEx(
            merged, cv2.MORPH_CLOSE, np.ones((ensure_odd(min(h, w) * 0.015), ensure_odd(min(h, w) * 0.015)), np.uint8)
        )
        merged = fill_holes(merged)
        merged = choose_best_component(merged)
        area_ratio = float(np.count_nonzero(merged)) / float(h * w)
        method = "grabcut"

    if area_ratio < 0.02 or area_ratio > 0.95:
        merged = np.full((h, w), 255, np.uint8)
        method = "full_frame_fallback"

    return merged, tuple(int(v) for v in bg_rgb), method


def bbox_from_mask(mask: np.ndarray, h: int, w: int, context_pad_ratio: float) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, w, h

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = int(round(max(x1 - x0, y1 - y0) * context_pad_ratio))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return x0, y0, x1, y1


def pad_to_square(
    image: Image.Image,
    mask: Image.Image,
    fill_color: tuple[int, int, int],
    max_square_side: int,
) -> tuple[Image.Image, Image.Image]:
    side = max(image.size)
    square_image = Image.new("RGB", (side, side), fill_color)
    square_mask = Image.new("L", (side, side), 0)
    offset = ((side - image.size[0]) // 2, (side - image.size[1]) // 2)
    square_image.paste(image, offset)
    square_mask.paste(mask, offset)

    if side > max_square_side:
        square_image = square_image.resize((max_square_side, max_square_side), Image.Resampling.LANCZOS)
        square_mask = square_mask.resize((max_square_side, max_square_side), Image.Resampling.NEAREST)

    return square_image, square_mask


def load_sample_records(source_root: Path) -> list[SampleRecord]:
    dataset_index = source_root / "dataset_index.csv"
    records: list[SampleRecord] = []
    with open(dataset_index, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record_path = source_root / row["record_path"]
            with open(record_path, "r", encoding="utf-8") as rec_handle:
                payload = json.load(rec_handle)
            records.append(
                SampleRecord(
                    sample_id=row["sample_id"],
                    class_name=row["class_name"],
                    source_dataset=row["source_dataset"],
                    source_code=row["source_code"],
                    source_object_id=str(row["source_object_id"]),
                    image_path=source_root / row["image_path"],
                    record_path=record_path,
                    image_url=payload.get("image_url"),
                    object_url=payload.get("object_url"),
                )
            )
    return records


def process_single_image(task: tuple[SampleRecord, str, int, float, int]) -> dict[str, object]:
    sample, output_root_str, max_square_side, context_pad_ratio, jpeg_quality = task
    output_root = Path(output_root_str)
    processed_image_root = output_root / "images"
    processed_mask_root = output_root / "object_masks"
    with Image.open(sample.image_path) as img:
        rgb_image = ImageOps.exif_transpose(img).convert("RGB")
    rgb_np = np.asarray(rgb_image)
    height, width = rgb_np.shape[:2]
    foreground_mask, bg_color, method = detect_foreground_mask(rgb_np, use_grabcut_fallback=True)
    x0, y0, x1, y1 = bbox_from_mask(foreground_mask, height, width, context_pad_ratio)

    cropped_image = rgb_image.crop((x0, y0, x1, y1))
    cropped_mask = Image.fromarray(foreground_mask, mode="L").crop((x0, y0, x1, y1))
    square_image, square_mask = pad_to_square(
        cropped_image,
        cropped_mask,
        bg_color,
        max_square_side=max_square_side,
    )

    image_output = processed_image_root / sample.class_name / f"{sample.sample_id}.jpg"
    mask_output = processed_mask_root / sample.class_name / f"{sample.sample_id}.png"
    image_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)

    square_image.save(image_output, format="JPEG", quality=jpeg_quality, optimize=True)
    square_mask.save(mask_output, format="PNG")

    return {
        "sample_id": sample.sample_id,
        "class_name": sample.class_name,
        "source_dataset": sample.source_dataset,
        "source_code": sample.source_code,
        "source_object_id": sample.source_object_id,
        "image_url": sample.image_url,
        "object_url": sample.object_url,
        "original_image_path": str(sample.image_path),
        "processed_image_path": str(image_output),
        "object_mask_path": str(mask_output),
        "original_width": width,
        "original_height": height,
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_x1": x1,
        "crop_y1": y1,
        "processed_side": square_image.size[0],
        "background_r": bg_color[0],
        "background_g": bg_color[1],
        "background_b": bg_color[2],
        "crop_method": method,
    }


def process_images(
    samples: list[SampleRecord],
    output_root: Path,
    max_square_side: int,
    context_pad_ratio: float,
    jpeg_quality: int,
    workers: int,
) -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    tasks = [
        (sample, str(output_root), max_square_side, context_pad_ratio, jpeg_quality)
        for sample in samples
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            manifests.append(future.result())
            if idx % 250 == 0 or idx == len(samples):
                print(f"processed_images={idx}/{len(samples)}")

    manifests.sort(key=lambda item: str(item["sample_id"]))
    return manifests


def stable_group_key(entry: dict[str, object]) -> str:
    image_url = entry.get("image_url")
    if image_url:
        return f"url::{image_url}"
    object_url = entry.get("object_url")
    if object_url:
        return f"object::{object_url}"
    return f"sample::{entry['sample_id']}"


def split_grouped_entries(
    manifests: list[dict[str, object]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for entry in manifests:
        stratum = (str(entry["class_name"]), str(entry["source_dataset"]))
        by_stratum[stratum][stable_group_key(entry)].append(entry)

    split_lists = {"train": [], "val": [], "test": []}

    for stratum, group_map in sorted(by_stratum.items()):
        groups = list(group_map.values())
        rng.shuffle(groups)
        total_items = sum(len(group) for group in groups)
        if len(groups) == 1:
            split_lists["train"].extend(groups[0])
            continue

        target_val = int(round(total_items * val_ratio))
        target_test = int(round(total_items * test_ratio))
        if total_items >= 10 and len(groups) >= 3:
            target_val = max(1, target_val)
            target_test = max(1, target_test)

        counts = {"train": 0, "val": 0, "test": 0}
        assignments: list[tuple[str, list[dict[str, object]]]] = []
        for group in sorted(groups, key=len, reverse=True):
            candidates = []
            if counts["val"] < target_val:
                candidates.append(("val", target_val - counts["val"]))
            if counts["test"] < target_test:
                candidates.append(("test", target_test - counts["test"]))
            if candidates:
                split_name = max(candidates, key=lambda item: item[1])[0]
            else:
                split_name = "train"
            assignments.append((split_name, group))
            counts[split_name] += len(group)

        if target_val > 0 and counts["val"] == 0:
            for idx, (split_name, group) in enumerate(assignments):
                if split_name == "train" and len(group) <= max(2, target_val):
                    assignments[idx] = ("val", group)
                    counts["train"] -= len(group)
                    counts["val"] += len(group)
                    break
        if target_test > 0 and counts["test"] == 0:
            for idx, (split_name, group) in enumerate(assignments):
                if split_name == "train" and len(group) <= max(2, target_test):
                    assignments[idx] = ("test", group)
                    counts["train"] -= len(group)
                    counts["test"] += len(group)
                    break

        for split_name, group in assignments:
            split_lists[split_name].extend(group)

    for split_name in split_lists:
        split_lists[split_name].sort(key=lambda item: str(item["sample_id"]))
    return split_lists


def write_split_files(output_root: Path, splits: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    split_root = output_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}

    for split_name, entries in splits.items():
        flist_path = split_root / f"{split_name}.flist"
        with open(flist_path, "w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(f"{entry['processed_image_path']}\n")

        class_counts = Counter(str(entry["class_name"]) for entry in entries)
        source_counts = Counter(str(entry["source_dataset"]) for entry in entries)
        summary[split_name] = {
            "count": len(entries),
            "class_counts": dict(sorted(class_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "flist_path": str(flist_path),
        }

    return summary


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape
        return w // 4, h // 4, w * 3 // 4, h * 3 // 4
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def boundary_band(mask: np.ndarray, thickness: int) -> np.ndarray:
    kernel = np.ones((ensure_odd(thickness), ensure_odd(thickness)), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return cv2.subtract(dilated, eroded)


def random_point_from(mask: np.ndarray, rng: random.Random) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    idx = rng.randrange(len(xs))
    return int(xs[idx]), int(ys[idx])


def draw_irregular_blob(
    canvas: np.ndarray,
    center: tuple[int, int],
    radius: int,
    rng: random.Random,
    count_range: tuple[int, int],
) -> None:
    cx, cy = center
    count = rng.randint(count_range[0], count_range[1])
    for _ in range(count):
        jitter_x = rng.randint(-radius, radius)
        jitter_y = rng.randint(-radius, radius)
        ax = rng.randint(max(4, radius // 3), max(5, radius))
        ay = rng.randint(max(4, radius // 3), max(5, radius))
        angle = rng.randint(0, 179)
        cv2.ellipse(
            canvas,
            (cx + jitter_x, cy + jitter_y),
            (ax, ay),
            angle,
            0,
            360,
            255,
            -1,
        )


def make_edge_chip_mask(obj_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = obj_mask.shape
    band = boundary_band(obj_mask, max(5, min(h, w) // 40))
    center = random_point_from(band, rng) or random_point_from(obj_mask, rng)
    if center is None:
        return np.zeros_like(obj_mask)

    mask = np.zeros_like(obj_mask)
    radius = max(12, min(h, w) // rng.randint(12, 22))
    draw_irregular_blob(mask, center, radius, rng, (3, 8))
    local_band = cv2.dilate(band, np.ones((ensure_odd(radius // 3), ensure_odd(radius // 3)), np.uint8), iterations=1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_or(local_band, obj_mask))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def make_inner_hole_mask(obj_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = obj_mask.shape
    center = random_point_from(obj_mask, rng)
    if center is None:
        return np.zeros_like(obj_mask)
    mask = np.zeros_like(obj_mask)
    radius = max(10, min(h, w) // rng.randint(14, 28))
    draw_irregular_blob(mask, center, radius, rng, (4, 10))
    mask = cv2.bitwise_and(mask, obj_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def make_fracture_mask(obj_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = obj_mask.shape
    points = []
    for _ in range(rng.randint(3, 6)):
        point = random_point_from(obj_mask, rng)
        if point is None:
            return np.zeros_like(obj_mask)
        points.append(point)
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    mask = np.zeros_like(obj_mask)
    thickness = max(3, min(h, w) // rng.randint(85, 120))
    cv2.polylines(mask, [pts], False, 255, thickness=thickness)
    if rng.random() < 0.75:
        hole_center = random_point_from(mask, rng)
        if hole_center is not None:
            draw_irregular_blob(mask, hole_center, max(8, thickness * 3), rng, (2, 4))
    mask = cv2.bitwise_and(mask, obj_mask)
    return mask


def make_part_missing_mask(obj_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = obj_mask.shape
    x0, y0, x1, y1 = mask_bbox(obj_mask)
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    side = rng.choice(["left", "right", "top", "bottom"])
    mask = np.zeros_like(obj_mask)

    if side in {"left", "right"}:
        anchor_y = rng.randint(y0, y1 - 1)
        anchor_x = x0 if side == "left" else x1 - 1
        width = max(10, bw // rng.randint(5, 9))
        height = max(18, bh // rng.randint(3, 6))
        direction = -1 if side == "left" else 1
        poly = np.array(
            [
                (anchor_x, anchor_y - height // 2),
                (anchor_x + direction * width, anchor_y - height // 3),
                (anchor_x + direction * width, anchor_y + height // 3),
                (anchor_x, anchor_y + height // 2),
            ],
            dtype=np.int32,
        )
    else:
        anchor_x = rng.randint(x0, x1 - 1)
        anchor_y = y0 if side == "top" else y1 - 1
        width = max(18, bw // rng.randint(3, 6))
        height = max(10, bh // rng.randint(5, 9))
        direction = -1 if side == "top" else 1
        poly = np.array(
            [
                (anchor_x - width // 2, anchor_y),
                (anchor_x - width // 3, anchor_y + direction * height),
                (anchor_x + width // 3, anchor_y + direction * height),
                (anchor_x + width // 2, anchor_y),
            ],
            dtype=np.int32,
        )

    cv2.fillPoly(mask, [poly], 255)
    mask = cv2.bitwise_and(mask, cv2.dilate(obj_mask, np.ones((11, 11), np.uint8), iterations=1))
    if rng.random() < 0.7:
        chip = make_edge_chip_mask(obj_mask, rng)
        mask = cv2.bitwise_or(mask, chip)
    return mask


def make_safe_center_mask(obj_mask: np.ndarray) -> np.ndarray:
    h, w = obj_mask.shape
    ys, xs = np.where(obj_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        mask = np.zeros_like(obj_mask)
        cv2.ellipse(mask, (w // 2, h // 2), (max(12, w // 10), max(12, h // 10)), 0, 0, 360, 255, -1)
        return mask

    x0, y0, x1, y1 = mask_bbox(obj_mask)
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = int(np.median(xs))
    cy = int(np.median(ys))
    rx = max(12, bw // 8)
    ry = max(12, bh // 8)

    mask = np.zeros_like(obj_mask)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    mask = cv2.bitwise_and(mask, obj_mask)
    if np.count_nonzero(mask) > 0:
        return mask

    anchor_idx = len(xs) // 2
    anchor = (int(xs[anchor_idx]), int(ys[anchor_idx]))
    mask = np.zeros_like(obj_mask)
    cv2.circle(mask, anchor, max(10, min(bw, bh) // 10), 255, -1)
    mask = cv2.bitwise_and(mask, obj_mask)
    if np.count_nonzero(mask) > 0:
        return mask

    mask = np.zeros_like(obj_mask)
    mask[anchor[1], anchor[0]] = 255
    return mask


def make_target_ratio_region_mask(
    obj_mask: np.ndarray,
    rng: random.Random,
    image_range: tuple[float, float],
) -> np.ndarray:
    h, w = obj_mask.shape
    ys, xs = np.where(obj_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = int(np.median(xs)), int(np.median(ys))

    target_ratio = rng.uniform(*image_range)
    target_area = max(1, int(round(target_ratio * h * w)))
    aspect = rng.uniform(0.6, 1.8)
    width = max(16, min(w, int(round(math.sqrt(target_area * aspect)))))
    height = max(16, min(h, int(round(target_area / max(1, width)))))
    width = min(w, max(16, width))
    height = min(h, max(16, height))

    jitter_x = int((rng.random() * 2 - 1) * max(8, width * 0.1))
    jitter_y = int((rng.random() * 2 - 1) * max(8, height * 0.1))
    x0 = max(0, min(w - width, cx - width // 2 + jitter_x))
    y0 = max(0, min(h - height, cy - height // 2 + jitter_y))
    x1 = min(w, x0 + width)
    y1 = min(h, y0 + height)

    mask = np.zeros_like(obj_mask)
    cv2.rectangle(mask, (x0, y0), (x1 - 1, y1 - 1), 255, -1)
    return mask


BUCKET_SPECS = {
    "bucket_0p01pct_20pct": {
        "name": "0.01%-20%",
        "slug": "0p01pct_20pct",
        "image_range": (0.0001, 0.2),
        "object_range": (0.02, 0.65),
        "allowed_region": "expanded_bbox",
        "source_min_object_ratio": 0.06,
        "strategies": [
            (("edge_chip", make_edge_chip_mask), 0.35),
            (("inner_hole", make_inner_hole_mask), 0.30),
            (("fracture", make_fracture_mask), 0.20),
            (("part_missing", make_part_missing_mask), 0.15),
        ],
    },
    "bucket_20pct_40pct": {
        "name": "20%-40%",
        "slug": "20pct_40pct",
        "image_range": (0.2, 0.4),
        "object_range": (0.15, 0.95),
        "allowed_region": "expanded_bbox_large",
        "source_min_object_ratio": 0.18,
        "strategies": [
            (("part_missing", make_part_missing_mask), 0.35),
            (("fracture", make_fracture_mask), 0.25),
            (("edge_chip", make_edge_chip_mask), 0.15),
            (("inner_hole", make_inner_hole_mask), 0.10),
            (("broad_box", make_part_missing_mask), 0.15),
        ],
    },
    "bucket_40pct_60pct": {
        "name": "40%-60%",
        "slug": "40pct_60pct",
        "image_range": (0.4, 0.6),
        "object_range": (0.20, 1.00),
        "allowed_region": "full_image",
        "source_min_object_ratio": 0.50,
        "strategies": [
            (("part_missing", make_part_missing_mask), 0.35),
            (("fracture", make_fracture_mask), 0.20),
            (("edge_chip", make_edge_chip_mask), 0.10),
            (("inner_hole", make_inner_hole_mask), 0.05),
            (("broad_box", make_part_missing_mask), 0.30),
        ],
    },
}

TRAIN_BUCKET_WEIGHTS = [
    ("bucket_0p01pct_20pct", 1.0 / 3.0),
    ("bucket_20pct_40pct", 1.0 / 3.0),
    ("bucket_40pct_60pct", 1.0 / 3.0),
]
FIXED_EVAL_BUCKET_ORDER = (
    "bucket_0p01pct_20pct",
    "bucket_20pct_40pct",
    "bucket_40pct_60pct",
)


def pick_weighted(rng: random.Random, weighted_items: list[tuple[object, float]]) -> object:
    pick = rng.random()
    cursor = 0.0
    for item, weight in weighted_items:
        cursor += weight
        if pick <= cursor:
            return item
    return weighted_items[-1][0]


def mask_ratios(mask: np.ndarray, obj_mask: np.ndarray) -> tuple[float, float]:
    image_pixels = float(mask.shape[0] * mask.shape[1])
    object_pixels = float(max(1, np.count_nonzero(obj_mask)))
    image_ratio = float(np.count_nonzero(mask)) / image_pixels
    overlap_pixels = float(np.count_nonzero(cv2.bitwise_and(mask, obj_mask)))
    object_ratio = overlap_pixels / object_pixels
    return image_ratio, object_ratio


def allowed_mask_region(obj_mask: np.ndarray, scope: str = "object") -> np.ndarray:
    h, w = obj_mask.shape
    if scope == "full_image":
        return np.full_like(obj_mask, 255)

    x0, y0, x1, y1 = mask_bbox(obj_mask)
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    if scope == "expanded_bbox":
        pad_x = max(12, int(round(bw * 0.30)))
        pad_y = max(12, int(round(bh * 0.30)))
        region = np.zeros_like(obj_mask)
        cv2.rectangle(
            region,
            (max(0, x0 - pad_x), max(0, y0 - pad_y)),
            (min(w - 1, x1 + pad_x - 1), min(h - 1, y1 + pad_y - 1)),
            255,
            -1,
        )
        return region

    if scope == "expanded_bbox_large":
        pad_x = max(18, int(round(bw * 0.60)))
        pad_y = max(18, int(round(bh * 0.60)))
        region = np.zeros_like(obj_mask)
        cv2.rectangle(
            region,
            (max(0, x0 - pad_x), max(0, y0 - pad_y)),
            (min(w - 1, x1 + pad_x - 1), min(h - 1, y1 + pad_y - 1)),
            255,
            -1,
        )
        return region

    margin = ensure_odd(max(7, min(h, w) // 40))
    return cv2.dilate(obj_mask, np.ones((margin, margin), np.uint8), iterations=1)


def regularize_mask(mask: np.ndarray, obj_mask: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
    if allowed is None:
        allowed = allowed_mask_region(obj_mask)
    mask = cv2.bitwise_and(mask, allowed)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def rescale_mask_to_range(
    mask: np.ndarray,
    obj_mask: np.ndarray,
    image_range: tuple[float, float],
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    h, w = mask.shape
    if allowed is None:
        allowed = allowed_mask_region(obj_mask)
    mask = regularize_mask(mask, obj_mask, allowed=allowed)

    grow_kernel = np.ones((ensure_odd(max(7, min(h, w) // 80)), ensure_odd(max(7, min(h, w) // 80))), np.uint8)
    shrink_kernel = np.ones((ensure_odd(max(5, min(h, w) // 120)), ensure_odd(max(5, min(h, w) // 120))), np.uint8)

    for _ in range(24):
        image_ratio, _ = mask_ratios(mask, obj_mask)
        if image_range[0] <= image_ratio <= image_range[1]:
            break
        if image_ratio < image_range[0]:
            grown = cv2.dilate(mask, grow_kernel, iterations=1)
            grown = cv2.bitwise_and(grown, allowed)
            if np.count_nonzero(grown) <= np.count_nonzero(mask):
                break
            mask = grown
        else:
            shrunk = cv2.erode(mask, shrink_kernel, iterations=1)
            if np.count_nonzero(shrunk) == 0:
                break
            mask = shrunk

    return regularize_mask(mask, obj_mask, allowed=allowed)


def build_mask_from_object(
    obj_mask: np.ndarray,
    rng: random.Random,
    bucket_key: str | None = None,
) -> tuple[np.ndarray, str, str, float, float]:
    if bucket_key is None:
        bucket_key = pick_weighted(rng, TRAIN_BUCKET_WEIGHTS)

    spec = BUCKET_SPECS[bucket_key]
    image_range = spec["image_range"]
    object_range = spec["object_range"]
    strategy_pool = spec["strategies"]
    allowed = allowed_mask_region(obj_mask, spec.get("allowed_region", "object"))

    for _ in range(48):
        chosen_name, chosen_fn = pick_weighted(rng, strategy_pool)
        mask = chosen_fn(obj_mask, rng)
        mask = rescale_mask_to_range(mask, obj_mask, image_range, allowed=allowed)
        image_ratio, object_ratio = mask_ratios(mask, obj_mask)
        if np.count_nonzero(mask) > 0 and image_range[0] <= image_ratio <= image_range[1] and object_range[0] <= object_ratio <= object_range[1]:
            return mask, chosen_name, bucket_key, image_ratio, object_ratio

    fallback = make_safe_center_mask(obj_mask)
    fallback = rescale_mask_to_range(fallback, obj_mask, image_range, allowed=allowed)
    image_ratio, object_ratio = mask_ratios(fallback, obj_mask)
    if image_range[0] <= image_ratio <= image_range[1]:
        return fallback, "safe_center", bucket_key, image_ratio, object_ratio

    target_region = make_target_ratio_region_mask(obj_mask, rng, image_range)
    target_region = rescale_mask_to_range(target_region, obj_mask, image_range, allowed=allowed)
    image_ratio, object_ratio = mask_ratios(target_region, obj_mask)
    if np.count_nonzero(target_region) > 0 and image_range[0] <= image_ratio <= image_range[1]:
        return target_region, "target_ratio_region", bucket_key, image_ratio, object_ratio

    return fallback, "safe_center", bucket_key, image_ratio, object_ratio


def build_eval_bucket_schedule(count: int, rng: random.Random) -> list[str]:
    schedule: list[str] = []
    while len(schedule) < count:
        schedule.extend(FIXED_EVAL_BUCKET_ORDER)
    schedule = schedule[:count]
    rng.shuffle(schedule)
    return schedule


def generate_mask_sample(task: tuple[int, str, str, int, str]) -> tuple[int, str, str, str, float, float]:
    idx, source_path_str, out_path_str, seed, bucket_key = task
    source_path = Path(source_path_str)
    out_path = Path(out_path_str)
    rng = random.Random(seed)
    obj_mask = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if obj_mask is None:
        raise RuntimeError(f"Failed to read object mask: {source_path}")
    obj_mask = np.where(obj_mask > 0, 255, 0).astype(np.uint8)
    mask, strategy_name, actual_bucket, image_ratio, object_ratio = build_mask_from_object(
        obj_mask,
        rng,
        bucket_key=bucket_key,
    )
    if np.count_nonzero(mask) == 0 or image_ratio < 0.003:
        mask = make_safe_center_mask(obj_mask)
        strategy_name = "fallback_safe_center"
        actual_bucket = bucket_key
        image_ratio, object_ratio = mask_ratios(mask, obj_mask)
    cv2.imwrite(str(out_path), mask)
    return idx, strategy_name, str(out_path), actual_bucket, image_ratio, object_ratio


def allocate_counts(total_count: int, weighted_items: list[tuple[str, float]]) -> dict[str, int]:
    counts = {name: int(total_count * weight) for name, weight in weighted_items}
    remainder = total_count - sum(counts.values())
    ordered = sorted(weighted_items, key=lambda item: item[1], reverse=True)
    idx = 0
    while remainder > 0:
        key = ordered[idx % len(ordered)][0]
        counts[key] += 1
        remainder -= 1
        idx += 1
    return counts


def write_mask_dataset(
    output_root: Path,
    manifests_by_split: dict[str, list[dict[str, object]]],
    train_mask_count: int,
    seed: int,
    workers: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    mask_root = output_root / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    eval_dir = mask_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_mask_sources = [Path(str(entry["object_mask_path"])) for entry in manifests_by_split["train"]]
    if not train_mask_sources:
        raise RuntimeError("No train object masks available to build HeriCera masks")

    source_coverages: dict[Path, float] = {}
    for path in train_mask_sources:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        source_coverages[path] = float((arr > 0).mean())

    bucket_source_paths: dict[str, list[Path]] = {}
    for bucket_key, bucket_spec in BUCKET_SPECS.items():
        min_ratio = float(bucket_spec["source_min_object_ratio"])
        eligible = [path for path, ratio in source_coverages.items() if ratio >= min_ratio]
        if len(eligible) < len(manifests_by_split["val"]):
            raise RuntimeError(
                f"Not enough eligible object masks for {bucket_spec['name']}: "
                f"need at least {len(manifests_by_split['val'])}, got {len(eligible)}"
            )
        bucket_source_paths[bucket_key] = eligible

    def run_mask_job(
        dataset_name: str,
        out_dir: Path,
        source_paths: list[Path],
        target_count: int,
        base_seed: int,
        bucket_schedule: list[str],
    ) -> dict[str, object]:
        flist_path = mask_root / f"{dataset_name}.flist"
        output_paths = [""] * target_count
        strategy_counts = Counter()
        bucket_counts = Counter()
        image_ratios: list[float] = []
        object_ratios: list[float] = []
        tasks = [
            (
                idx,
                str(source_paths[idx]),
                str(out_dir / f"{dataset_name}_mask_{idx:05d}.png"),
                base_seed + idx,
                bucket_schedule[idx],
            )
            for idx in range(target_count)
        ]

        if workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                iterator = executor.map(generate_mask_sample, tasks)
                for done_count, (idx, strategy_name, out_path, actual_bucket, image_ratio, object_ratio) in enumerate(iterator, start=1):
                    strategy_counts[f"{actual_bucket}_{strategy_name}"] += 1
                    bucket_counts[actual_bucket] += 1
                    image_ratios.append(float(image_ratio))
                    object_ratios.append(float(object_ratio))
                    output_paths[idx] = out_path
                    if done_count % 500 == 0 or done_count == target_count:
                        print(f"generated_{dataset_name}_masks={done_count}/{target_count}")
        else:
            for done_count, (idx, strategy_name, out_path, actual_bucket, image_ratio, object_ratio) in enumerate(map(generate_mask_sample, tasks), start=1):
                strategy_counts[f"{actual_bucket}_{strategy_name}"] += 1
                bucket_counts[actual_bucket] += 1
                image_ratios.append(float(image_ratio))
                object_ratios.append(float(object_ratio))
                output_paths[idx] = out_path
                if done_count % 500 == 0 or done_count == target_count:
                    print(f"generated_{dataset_name}_masks={done_count}/{target_count}")

        with open(flist_path, "w", encoding="utf-8") as handle:
            for path in output_paths:
                handle.write(f"{path}\n")

        image_ratios_arr = np.array(image_ratios, dtype=np.float64)
        object_ratios_arr = np.array(object_ratios, dtype=np.float64)
        return {
            "count": target_count,
            "flist_path": str(flist_path),
            "strategy_counts": dict(sorted(strategy_counts.items())),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "image_ratio_min": float(image_ratios_arr.min()),
            "image_ratio_p25": float(np.quantile(image_ratios_arr, 0.25)),
            "image_ratio_mean": float(image_ratios_arr.mean()),
            "image_ratio_p75": float(np.quantile(image_ratios_arr, 0.75)),
            "image_ratio_max": float(image_ratios_arr.max()),
            "object_ratio_min": float(object_ratios_arr.min()),
            "object_ratio_mean": float(object_ratios_arr.mean()),
            "object_ratio_max": float(object_ratios_arr.max()),
        }

    summaries: dict[str, object] = {}
    train_bucket_counts = allocate_counts(train_mask_count, TRAIN_BUCKET_WEIGHTS)
    combined_train_paths: list[str] = []
    train_combined_flist = mask_root / "train.flist"

    for bucket_index, (bucket_key, bucket_spec) in enumerate(BUCKET_SPECS.items()):
        count = train_bucket_counts[bucket_key]
        bucket_dir = mask_root / f"train_{bucket_spec['slug']}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        source_paths = list(bucket_source_paths[bucket_key])
        rng_local = random.Random(seed + 1000 + bucket_index)
        rng_local.shuffle(source_paths)
        dataset_name = f"train_bucket_{bucket_spec['slug']}"
        summaries[dataset_name] = run_mask_job(
            dataset_name=dataset_name,
            out_dir=bucket_dir,
            source_paths=[source_paths[idx % len(source_paths)] for idx in range(count)],
            target_count=count,
            base_seed=seed + 5000 * (bucket_index + 1),
            bucket_schedule=[bucket_key] * count,
        )
        bucket_paths = [
            line.strip()
            for line in Path(summaries[dataset_name]["flist_path"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        combined_train_paths.extend(bucket_paths)

    rng.shuffle(combined_train_paths)
    with open(train_combined_flist, "w", encoding="utf-8") as handle:
        for path in combined_train_paths:
            handle.write(f"{path}\n")

    summaries["train"] = {
        "count": len(combined_train_paths),
        "flist_path": str(train_combined_flist),
        "bucket_counts": {
            BUCKET_SPECS[key]["name"]: train_bucket_counts[key] for key in BUCKET_SPECS
        },
        "bucket_flists": {
            BUCKET_SPECS[key]["name"]: summaries[f"train_bucket_{BUCKET_SPECS[key]['slug']}"]["flist_path"]
            for key in BUCKET_SPECS
        },
    }

    eval_specs = [
        ("eval_val", "val", seed + 100000),
        ("eval_test", "test", seed + 200000),
    ]

    for dataset_name, split_name, base_seed in eval_specs:
        target_count = len(manifests_by_split[split_name])
        if target_count <= 0:
            raise RuntimeError(f"No samples found in split: {split_name}")

        bucket_flist_map: dict[str, str] = {}
        bucket_count_map: dict[str, int] = {}
        for bucket_index, bucket_key in enumerate(FIXED_EVAL_BUCKET_ORDER):
            bucket_spec = BUCKET_SPECS[bucket_key]
            bucket_dir = mask_root / f"{dataset_name}_{bucket_spec['slug']}"
            bucket_dir.mkdir(parents=True, exist_ok=True)
            source_paths = list(bucket_source_paths[bucket_key])
            rng_local = random.Random(base_seed + bucket_index)
            rng_local.shuffle(source_paths)
            bucket_dataset_name = f"{dataset_name}_bucket_{bucket_spec['slug']}"
            summaries[bucket_dataset_name] = run_mask_job(
                dataset_name=bucket_dataset_name,
                out_dir=bucket_dir,
                source_paths=source_paths[:target_count],
                target_count=target_count,
                base_seed=base_seed + 10000 * (bucket_index + 1),
                bucket_schedule=[bucket_key] * target_count,
            )
            bucket_flist_map[bucket_spec["name"]] = summaries[bucket_dataset_name]["flist_path"]
            bucket_count_map[bucket_spec["name"]] = target_count

        primary_bucket = BUCKET_SPECS[FIXED_EVAL_BUCKET_ORDER[0]]
        primary_dataset_name = f"{dataset_name}_bucket_{primary_bucket['slug']}"
        shutil.copy2(Path(summaries[primary_dataset_name]["flist_path"]), mask_root / f"{dataset_name}.flist")

        alias_dir = eval_dir if dataset_name == "eval_test" else (mask_root / dataset_name)
        alias_dir.mkdir(parents=True, exist_ok=True)
        src_dir = mask_root / f"{dataset_name}_{primary_bucket['slug']}"
        for src_path in sorted(src_dir.glob("*.png")):
            shutil.copy2(src_path, alias_dir / src_path.name)

        summaries[dataset_name] = {
            "count": target_count,
            "flist_path": str(mask_root / f"{dataset_name}.flist"),
            "bucket_counts": bucket_count_map,
            "bucket_flists": bucket_flist_map,
            "alias_of": primary_bucket["name"],
        }

    shutil.copy2(mask_root / "eval_test.flist", mask_root / "eval.flist")
    for src_path in sorted((mask_root / "eval_test").glob("*.png")):
        shutil.copy2(src_path, eval_dir / src_path.name)
    summaries["eval"] = {
        "count": summaries["eval_test"]["count"],
        "flist_path": str(mask_root / "eval.flist"),
        "alias_of": summaries["eval_test"]["alias_of"],
        "bucket_counts": summaries["eval_test"]["bucket_counts"],
        "bucket_flists": summaries["eval_test"]["bucket_flists"],
    }

    return summaries


def create_preview_strip(
    manifests: list[dict[str, object]],
    output_root: Path,
    preview_count: int,
    seed: int,
) -> None:
    if preview_count <= 0 or not manifests:
        return

    preview_dir = output_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    picked = manifests.copy()
    rng.shuffle(picked)
    picked = picked[: min(preview_count, len(picked))]

    tiles = []
    for entry in picked:
        original = Image.open(entry["original_image_path"]).convert("RGB")
        processed = Image.open(entry["processed_image_path"]).convert("RGB")
        object_mask = Image.open(entry["object_mask_path"]).convert("L")

        x0, y0, x1, y1 = (
            int(entry["crop_x0"]),
            int(entry["crop_y0"]),
            int(entry["crop_x1"]),
            int(entry["crop_y1"]),
        )
        original_draw = original.copy()
        draw = ImageDraw.Draw(original_draw)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 0, 0), width=6)

        tile_size = 320
        mask_rgb = Image.merge("RGB", (object_mask, object_mask, object_mask))
        panels = [
            ImageOps.contain(original_draw, (tile_size, tile_size), Image.Resampling.LANCZOS),
            ImageOps.contain(processed, (tile_size, tile_size), Image.Resampling.LANCZOS),
            ImageOps.contain(mask_rgb, (tile_size, tile_size), Image.Resampling.NEAREST),
        ]

        strip = Image.new("RGB", (tile_size * 3, tile_size), (250, 250, 250))
        for idx, panel in enumerate(panels):
            x = idx * tile_size + (tile_size - panel.size[0]) // 2
            y = (tile_size - panel.size[1]) // 2
            strip.paste(panel, (x, y))
        tiles.append(strip)

    rows = []
    per_row = 2
    for start in range(0, len(tiles), per_row):
        row_tiles = tiles[start : start + per_row]
        width = sum(tile.size[0] for tile in row_tiles)
        height = max(tile.size[1] for tile in row_tiles)
        row = Image.new("RGB", (width, height), (255, 255, 255))
        x = 0
        for tile in row_tiles:
            row.paste(tile, (x, 0))
            x += tile.size[0]
        rows.append(row)

    if not rows:
        return

    preview = Image.new(
        "RGB",
        (max(row.size[0] for row in rows), sum(row.size[1] for row in rows)),
        (255, 255, 255),
    )
    y = 0
    for row in rows:
        preview.paste(row, (0, y))
        y += row.size[1]
    preview.save(preview_dir / "preprocess_preview.jpg", quality=95)


def write_manifest(output_root: Path, manifests: list[dict[str, object]]) -> Path:
    manifest_path = output_root / "manifest.csv"
    fieldnames = list(manifests[0].keys()) if manifests else []
    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifests)
    return manifest_path


def write_readme(
    output_root: Path,
    source_root: Path,
    manifest_path: Path,
    split_summary: dict[str, object],
    mask_summary: dict[str, object],
    args: argparse.Namespace,
) -> None:
    readme = f"""# HeriCera-6C PATINA Training Pack

This directory contains a training-ready derivative of `HeriCera-6C` prepared for PATINA image inpainting experiments.

## What Changed

- Source root: `{source_root}`
- Object-preserving crop: enabled
- Square export policy: crop around detected object, pad to square, cap the square side at `{args.max_square_side}`
- Output images: JPEG
- Object masks: saved per sample for mask-aware data generation
- Split policy: grouped by shared `image_url`/`object_url` and stratified by `class_name + source_dataset`
- Ceramic mask policy: HeriCera-only masks generated from train-split object masks
- Evaluation policy: paper-style three buckets with fixed one-to-one pairing per bucket

## Split Sizes

- `train`: {split_summary['train']['count']}
- `val`: {split_summary['val']['count']}
- `test`: {split_summary['test']['count']}

## Mask Pools

- `train`: {mask_summary['train']['count']}
- `eval_val`: {mask_summary['eval_val']['count']} per bucket
- `eval_test`: {mask_summary['eval_test']['count']} per bucket

## Files

- `images/`: processed square training images grouped by class
- `object_masks/`: foreground object masks aligned with processed images
- `splits/`: `train.flist`, `val.flist`, `test.flist`
- `masks/train.flist`: combined three-bucket training mask pool
- `masks/train_bucket_*.flist`: bucket-specific training mask pools
- `masks/eval_val_bucket_*.flist`: fixed validation bucket masks
- `masks/eval_test_bucket_*.flist`: fixed held-out test bucket masks
- `masks/eval_val.flist`: backward-compatible alias to the `0.01%-20%` validation bucket
- `masks/eval_test.flist`: backward-compatible alias to the `0.01%-20%` test bucket
- `masks/eval.flist`: backward-compatible alias to `eval_test.flist`
- `manifest.csv`: per-sample preprocessing manifest
- `previews/preprocess_preview.jpg`: quick visual sanity check

## Training Notes

- Use `splits/train.flist` for `TRAIN_INPAINT_IMAGE_FLIST`
- Use `splits/val.flist` with `masks/eval_val_bucket_*.flist` for bucketed validation
- Use `splits/test.flist` with `masks/eval_test_bucket_*.flist` for held-out reporting
- Use `masks/train.flist` for `TRAIN_MASK_FLIST`
- Default protocol uses only HeriCera-derived masks; CelebA-HQ masks are excluded from evaluation.
- Recommended schedule: train at `256`, then continue fine-tuning at `512`

## Manifest

- `{manifest_path}`
"""
    with open(output_root / "README.md", "w", encoding="utf-8") as handle:
        handle.write(readme)


def write_summary_json(
    output_root: Path,
    manifests: list[dict[str, object]],
    split_summary: dict[str, object],
    mask_summary: dict[str, object],
    args: argparse.Namespace,
) -> None:
    crop_method_counts = Counter(str(entry["crop_method"]) for entry in manifests)
    processed_side_counts = Counter(int(entry["processed_side"]) for entry in manifests)
    source_counts = Counter(str(entry["source_dataset"]) for entry in manifests)
    class_counts = Counter(str(entry["class_name"]) for entry in manifests)
    summary = {
        "pack_name": "hericera6c_semnet",
        "source_dataset_name": "HeriCera-6C",
        "source_root": str(args.source_root),
        "output_root": str(output_root),
        "total_images": len(manifests),
        "class_counts": dict(sorted(class_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "crop_method_counts": dict(sorted(crop_method_counts.items())),
        "processed_side_counts": dict(sorted(processed_side_counts.items())),
        "max_square_side": args.max_square_side,
        "context_pad_ratio": args.context_pad_ratio,
        "split_summary": split_summary,
        "mask_summary": mask_summary,
        "evaluation_protocol": {
            "train_mask_source": "train split object_masks only",
            "eval_mask_source": "train split object_masks only",
            "eval_pairing": {
                "val": "one_to_one_fixed_per_bucket",
                "test": "one_to_one_fixed_per_bucket",
            },
            "bucket_names": [spec["name"] for spec in BUCKET_SPECS.values()],
            "cross_domain_masks": False,
            "generic_eval_alias": "masks/eval.flist -> eval_test.flist -> 0.01%-20% bucket",
        },
        "seed": args.seed,
    }
    with open(output_root / "pack_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def maybe_remove_existing(output_root: Path, overwrite: bool) -> None:
    if not output_root.exists():
        return
    if not overwrite:
        raise RuntimeError(f"Output directory already exists: {output_root}")
    shutil.rmtree(output_root)


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    maybe_remove_existing(args.output_root, args.overwrite)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"loading_records_from={args.source_root}")
    samples = load_sample_records(args.source_root)
    print(f"total_samples={len(samples)}")

    manifests = process_images(
        samples=samples,
        output_root=args.output_root,
        max_square_side=args.max_square_side,
        context_pad_ratio=args.context_pad_ratio,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
    )

    manifest_path = write_manifest(args.output_root, manifests)
    splits = split_grouped_entries(
        manifests=manifests,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    split_summary = write_split_files(args.output_root, splits)
    mask_summary = write_mask_dataset(
        output_root=args.output_root,
        manifests_by_split=splits,
        train_mask_count=args.train_mask_count,
        seed=args.seed,
        workers=args.workers,
    )
    create_preview_strip(
        manifests=manifests,
        output_root=args.output_root,
        preview_count=args.preview_count,
        seed=args.seed,
    )
    write_readme(
        output_root=args.output_root,
        source_root=args.source_root,
        manifest_path=manifest_path,
        split_summary=split_summary,
        mask_summary=mask_summary,
        args=args,
    )
    write_summary_json(
        output_root=args.output_root,
        manifests=manifests,
        split_summary=split_summary,
        mask_summary=mask_summary,
        args=args,
    )

    print(f"training_pack_ready={args.output_root}")


if __name__ == "__main__":
    main()
