import argparse
import json
import os

from cleanfid import fid


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def count_images(folder):
    return sum(
        1
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name)) and name.lower().endswith(VALID_EXTENSIONS)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True, help="Generated image directory")
    parser.add_argument("--gt_dir", required=True, help="Ground-truth image directory")
    parser.add_argument("--mode", default="clean", choices=["clean", "legacy_pytorch", "legacy_tensorflow"])
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    gt_count = count_images(args.gt_dir)
    gen_count = count_images(args.gen_dir)
    if gt_count < 2 or gen_count < 2:
        raise ValueError(
            f"FID requires at least 2 images per directory, got gt={gt_count}, gen={gen_count}."
        )

    score = fid.compute_fid(args.gt_dir, args.gen_dir, mode=args.mode)
    print(f"FID ({args.mode}): {score:.6f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "gt_dir": args.gt_dir,
                    "gen_dir": args.gen_dir,
                    "mode": args.mode,
                    "fid": score,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
