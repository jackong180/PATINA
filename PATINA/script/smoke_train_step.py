import argparse
from pathlib import Path
import sys

import torch
import torchvision.models as tv_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
import src.loss as loss_mod
from src.models import InpaintingModel


def build_config(config_path: Path, device: torch.device, save_dir: Path) -> Config:
    config = Config(str(config_path))
    config.MODE = 1
    config.MODEL = 2
    config.DEVICE = device
    config.GPU = [device.index or 0]
    config.PRETRAIN_FROM = None
    config.RESUME_FROM = None
    config.SAVE_HISTORY = 0
    config.AUTO_TEST_AFTER_TRAIN = 0
    config.ENABLE_LR_SCHEDULER = 0
    config.CHECKPOINTS_DIR = str(save_dir)
    return config


def make_batch(batch_size: int, image_size: int, device: torch.device):
    images = torch.rand(batch_size, 3, image_size, image_size, device=device)
    masks = (torch.rand(batch_size, 1, image_size, image_size, device=device) > 0.7).float()
    return images, masks


def main():
    parser = argparse.ArgumentParser(description="Run one GPU training smoke step for PATINA.")
    parser.add_argument("--config", type=str, default="checkpoints/config.yml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/smoke_train",
        help="directory used for temporary checkpoint outputs during the smoke step",
    )
    parser.add_argument(
        "--offline-vgg",
        action="store_true",
        help="avoid torchvision weight downloads by building VGG19 without pretrained weights",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("smoke_train_step.py is intended for CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the current environment")

    if args.offline_vgg:
        loss_mod._vgg19_features = lambda: tv_models.vgg19(weights=None).features

    config_path = (PROJECT_ROOT / args.config).resolve()
    save_dir = (PROJECT_ROOT / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(config_path, device, save_dir)
    model = InpaintingModel(config).to(device)
    model.train()

    images, masks = make_batch(args.batch_size, args.image_size, device)
    outputs, gen_loss, dis_loss, logs, *_ = model.process(images, masks)
    model.backward(gen_loss, dis_loss)
    torch.cuda.synchronize(device)

    print(
        {
            "device": str(device),
            "batch_size": int(args.batch_size),
            "image_size": int(args.image_size),
            "outputs_shape": tuple(outputs.shape),
            "gen_loss": float(gen_loss.detach().cpu()),
            "dis_loss": float(dis_loss.detach().cpu()),
            "logs": logs,
        }
    )


if __name__ == "__main__":
    main()
