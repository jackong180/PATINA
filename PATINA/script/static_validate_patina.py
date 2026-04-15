import argparse
import compileall
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    PROJECT_ROOT / "src" / "lcbc_module.py",
    PROJECT_ROOT / "src" / "mrda_module.py",
    PROJECT_ROOT / "src" / "dfcc_module.py",
)

REQUIRED_CONFIG_KEYS = (
    "LCBC_LATENT_ENABLE",
    "MRDA_STAGE1_ENABLE",
    "DFCC_LATENT_ENABLE",
)

FORBIDDEN_CONFIG_KEYS = (
    "CA_LATENT_ENABLE",
    "HINT_MPD_STAGE1_ENABLE",
    "UFFC_LATENT_ENABLE",
    "MGCTR_PRECONDITION_ENABLE",
    "MGCTR_MASK_ROUTE_ENABLE",
    "MGCTR_SKIP_FUSION_ENABLE",
    "MGCTR_REFINEMENT_ENABLE",
)

FORBIDDEN_SOURCE_TOKENS = (
    "contextual_attention_module",
    "hint_mpd_module",
    "uffc_module",
    "ContextualAttentionAdapter",
    "MaskAwarePixelShuffleDownsample",
    "UFFCResidualBlock",
    "SEM-Net-baseline",
    "smoke_pact",
    "smoke_mgctr",
    "MGCTR_",
)

SOURCE_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "main.py",
    PROJECT_ROOT / "script" / "smoke_patina.py",
    PROJECT_ROOT / "script" / "smoke_branch_ablation.py",
)


def ensure_required_files():
    missing = [str(path) for path in REQUIRED_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing renamed module files:\n" + "\n".join(missing))


def ensure_config_keys(config_path: Path):
    config_keys = set()
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        if ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0].strip()
        if key:
            config_keys.add(key)

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config_keys]
    if missing:
        raise KeyError("Missing PATINA config keys: " + ", ".join(missing))
    forbidden = [key for key in FORBIDDEN_CONFIG_KEYS if key in config_keys]
    if forbidden:
        raise KeyError("Legacy config keys still present: " + ", ".join(forbidden))


def ensure_no_legacy_source_tokens():
    offenders = []
    candidate_paths = []
    for root in SOURCE_ROOTS:
        if root.is_dir():
            candidate_paths.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
        elif root.is_file():
            candidate_paths.append(root)

    for path in candidate_paths:
        text = path.read_text(encoding="utf-8")
        hits = [token for token in FORBIDDEN_SOURCE_TOKENS if token in text]
        if hits:
            offenders.append((path, hits))

    if offenders:
        lines = []
        for path, hits in offenders:
            lines.append(f"{path}: {', '.join(hits)}")
        raise ValueError("Legacy module names still present in source files:\n" + "\n".join(lines))


def run_compileall():
    if not compileall.compile_dir(str(PROJECT_ROOT), quiet=1):
        raise RuntimeError("compileall reported syntax/import compilation failures")


def main():
    parser = argparse.ArgumentParser(description="Static validation for the renamed PATINA codebase.")
    parser.add_argument("--config", type=str, default="checkpoints/config.yml")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    ensure_required_files()
    ensure_config_keys(config_path)
    ensure_no_legacy_source_tokens()
    run_compileall()
    print("PATINA static validation passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI failure surface
        print(f"PATINA static validation failed: {exc}", file=sys.stderr)
        raise
