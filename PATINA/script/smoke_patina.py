import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.dataset import Dataset
from src.models import InpaintingModel


PRESETS = {
    "baseline_zero": {
        "lcbc": 0.0,
        "mrda_stage1": 0.0,
        "mrda_stage2": 0.0,
        "mrda_stage3": 0.0,
        "dfcc_latent": 0.0,
        "dfcc_decoder3": 0.0,
        "precondition_enable": 0,
        "mask_route_enable": 0,
        "skip_enable": 0,
        "refinement_enable": 0,
        "precondition_scale": 0.0,
        "route_scale": 0.0,
        "skip_scale": 0.0,
        "refinement_scale": 0.0,
    },
    "PATINA_default": {
        "lcbc": 0.03,
        "mrda_stage1": 0.00,
        "mrda_stage2": 0.06,
        "mrda_stage3": 0.10,
        "dfcc_latent": 0.04,
        "dfcc_decoder3": 0.02,
        "precondition_enable": 1,
        "mask_route_enable": 1,
        "skip_enable": 1,
        "refinement_enable": 1,
        "precondition_scale": 0.00,
        "route_scale": 0.005,
        "skip_scale": 0.02,
        "refinement_scale": 0.00,
    },
    "PATINA_route_tiny": {
        "lcbc": 0.03,
        "mrda_stage1": 0.00,
        "mrda_stage2": 0.06,
        "mrda_stage3": 0.10,
        "dfcc_latent": 0.04,
        "dfcc_decoder3": 0.02,
        "precondition_enable": 1,
        "mask_route_enable": 1,
        "skip_enable": 1,
        "refinement_enable": 1,
        "precondition_scale": 0.00,
        "route_scale": 0.01,
        "skip_scale": 0.02,
        "refinement_scale": 0.00,
    },
    "PATINA_route_micro": {
        "lcbc": 0.03,
        "mrda_stage1": 0.00,
        "mrda_stage2": 0.06,
        "mrda_stage3": 0.10,
        "dfcc_latent": 0.04,
        "dfcc_decoder3": 0.02,
        "precondition_enable": 1,
        "mask_route_enable": 1,
        "skip_enable": 1,
        "refinement_enable": 1,
        "precondition_scale": 0.00,
        "route_scale": 0.005,
        "skip_scale": 0.02,
        "refinement_scale": 0.00,
    },
    "PATINA_skip_tiny": {
        "lcbc": 0.03,
        "mrda_stage1": 0.00,
        "mrda_stage2": 0.06,
        "mrda_stage3": 0.10,
        "dfcc_latent": 0.04,
        "dfcc_decoder3": 0.02,
        "precondition_enable": 1,
        "mask_route_enable": 1,
        "skip_enable": 1,
        "refinement_enable": 1,
        "precondition_scale": 0.00,
        "route_scale": 0.02,
        "skip_scale": 0.01,
        "refinement_scale": 0.00,
    },
    "PATINA_no_deep_route": {
        "lcbc": 0.03,
        "mrda_stage1": 0.00,
        "mrda_stage2": 0.06,
        "mrda_stage3": 0.10,
        "dfcc_latent": 0.04,
        "dfcc_decoder3": 0.02,
        "precondition_enable": 0,
        "mask_route_enable": 0,
        "skip_enable": 1,
        "refinement_enable": 1,
        "precondition_scale": 0.00,
        "route_scale": 0.00,
        "skip_scale": 0.02,
        "refinement_scale": 0.00,
    },
    "PATINA_no_skip_adapter": {
        "lcbc": 0.03,
        "mrda_stage1": 0.00,
        "mrda_stage2": 0.06,
        "mrda_stage3": 0.10,
        "dfcc_latent": 0.04,
        "dfcc_decoder3": 0.02,
        "precondition_enable": 1,
        "mask_route_enable": 1,
        "skip_enable": 0,
        "refinement_enable": 1,
        "precondition_scale": 0.00,
        "route_scale": 0.02,
        "skip_scale": 0.00,
        "refinement_scale": 0.00,
    },
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(config_path, pretrain_path, device):
    config = Config(str(config_path))
    config.MODE = 2
    config.MODEL = 2
    config.MASK = 6
    config.PRETRAIN_FROM = str(pretrain_path)
    config.RESUME_FROM = None
    config.DEVICE = device
    config.GPU = [0] if device.type == "cuda" else []
    config.INPUT_SIZE = int(getattr(config, "INPUT_SIZE", 256) or 256)
    return config


def apply_config_preset(config, preset):
    config.PATINA_PRECONDITION_ENABLE = int(preset.get("precondition_enable", 0))
    config.PATINA_MASK_ROUTE_ENABLE = int(preset.get("mask_route_enable", 0))
    config.PATINA_SKIP_FUSION_ENABLE = int(preset.get("skip_enable", 0))
    config.PATINA_REFINEMENT_ENABLE = int(preset.get("refinement_enable", 0))
    config.PATINA_PRECONDITION_SCALE_INIT = float(preset.get("precondition_scale", 0.0))
    config.PATINA_MASK_ROUTE_SCALE_INIT = float(preset.get("route_scale", 0.0))
    config.PATINA_SKIP_ADAPTER_RES_SCALE_INIT = float(preset.get("skip_scale", 0.0))
    config.PATINA_REFINEMENT_RES_SCALE_INIT = float(preset.get("refinement_scale", 0.0))
    config.LCBC_LATENT_RES_SCALE_INIT = float(preset.get("lcbc", 0.0))
    config.MRDA_STAGE1_RES_SCALE_INIT = float(preset.get("mrda_stage1", 0.0))
    config.MRDA_STAGE2_RES_SCALE_INIT = float(preset.get("mrda_stage2", 0.0))
    config.MRDA_STAGE3_RES_SCALE_INIT = float(preset.get("mrda_stage3", 0.0))
    config.DFCC_LATENT_RES_SCALE_INIT = float(preset.get("dfcc_latent", 0.0))
    config.DFCC_DECODER3_RES_SCALE_INIT = float(preset.get("dfcc_decoder3", 0.0))


def unwrap_generator(generator):
    return generator.module if isinstance(generator, torch.nn.DataParallel) else generator


def set_raw_parameter(parameter, value):
    if parameter is not None:
        parameter.data.fill_(float(value))


def apply_preset(generator, preset):
    set_raw_parameter(
        getattr(getattr(generator, "lcbc_latent", None), "residual_scale", None),
        preset["lcbc"],
    )
    set_raw_parameter(getattr(generator.down1_2, "residual_scale", None), preset["mrda_stage1"])
    set_raw_parameter(getattr(generator.down2_3, "residual_scale", None), preset["mrda_stage2"])
    set_raw_parameter(getattr(generator.down3_4, "residual_scale", None), preset["mrda_stage3"])
    set_raw_parameter(
        getattr(getattr(generator, "dfcc_latent", None), "residual_scale", None),
        preset["dfcc_latent"],
    )
    set_raw_parameter(
        getattr(getattr(generator, "dfcc_decoder_level3", None), "residual_scale", None),
        preset["dfcc_decoder3"],
    )
    set_raw_parameter(getattr(generator.skip_adapter_level1, "residual_scale", None), preset["skip_scale"])
    set_raw_parameter(getattr(generator.skip_adapter_level2, "residual_scale", None), preset["skip_scale"])
    set_raw_parameter(getattr(generator.skip_adapter_level3, "residual_scale", None), preset["skip_scale"])
    set_raw_parameter(getattr(generator.refinement_head, "residual_scale", None), preset["refinement_scale"])

    for module in generator.modules():
        if hasattr(module, "pre_scale") and getattr(module, "pre_scale", None) is not None:
            set_raw_parameter(module.pre_scale, preset["precondition_scale"])
        if hasattr(module, "route_scale") and getattr(module, "route_scale", None) is not None:
            set_raw_parameter(module.route_scale, preset["route_scale"])


@torch.no_grad()
def eval_bucket(model, dataset, limit, device):
    subset = Subset(dataset, list(range(min(limit, len(dataset)))))
    loader = DataLoader(
        subset,
        batch_size=4 if device.type == "cuda" else 1,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    values = []
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))
        mask_tensor = masks.expand_as(outputs_merged)
        batch_abs = torch.abs(outputs_merged - images).mul(mask_tensor).flatten(1).sum(dim=1)
        batch_denom = mask_tensor.flatten(1).sum(dim=1)
        batch_scores = torch.where(batch_denom > 0, batch_abs / batch_denom, torch.zeros_like(batch_denom))
        values.extend(float(score) for score in batch_scores.cpu())
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description="Small validation smoke test for PATINA residual presets.")
    parser.add_argument("--config", type=str, default="checkpoints/config.yml")
    parser.add_argument(
        "--pretrain",
        type=str,
        default="checkpoints/InpaintingModel_gen.pth",
    )
    parser.add_argument("--samples-per-bucket", type=int, default=12)
    parser.add_argument("--output", type=str, default="smoke_patina_results.json")
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    pretrain_path = (PROJECT_ROOT / args.pretrain).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    results = []
    for preset_name, preset in PRESETS.items():
        set_seed(args.seed)
        config = build_config(config_path, pretrain_path, device)
        apply_config_preset(config, preset)
        model = InpaintingModel(config).to(device)
        model.load()
        model.eval()
        apply_preset(unwrap_generator(model.generator), preset)

        bucket_scores = {}
        for bucket in config.VAL_MASK_BUCKETS:
            dataset = Dataset(
                config,
                config.VAL_INPAINT_IMAGE_FLIST,
                bucket["mask_flist"],
                augment=False,
                training=False,
            )
            bucket_scores[bucket["name"]] = eval_bucket(
                model,
                dataset,
                args.samples_per_bucket,
                device,
            )

        results.append(
            {
                "preset": preset_name,
                "params": preset,
                "masked_l1_mean": float(np.mean(list(bucket_scores.values()))),
                "bucket_scores": bucket_scores,
                "samples_per_bucket": int(args.samples_per_bucket),
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results.sort(key=lambda item: item["masked_l1_mean"])
    payload = {
        "device": str(device),
        "config": str(config_path),
        "pretrain": str(pretrain_path),
        "results": results,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved smoke results to: {output_path}")


if __name__ == "__main__":
    main()
