# Contributing

This repository is maintained as the final reproducible code release for the PATINA manuscript.

## Scope

- Keep changes tightly aligned with ceramic artifact image inpainting
- Preserve reproducibility of the paper-aligned PATINA experiments
- Avoid introducing unrelated framework features that are not needed by this project

## Before Changing Core Code

Please review the paper-specific modules first:

- `src/mrda_module.py`
- `src/lcbc_module.py`
- `src/dfcc_module.py`
- `src/adaptive_fusion_module.py`
- `src/networks.py`

## Reproducibility Guidelines

- Use a Python 3.10 environment with CUDA-enabled PyTorch for paper-aligned runs unless you are intentionally validating a different stack
- Keep dataset naming aligned with the ceramic artifact restoration setup used by PATINA
- If you modify training defaults, make sure the change does not alter the intended paper protocol unintentionally
- Prefer adding new scripts under `script/` rather than overloading the public entrypoints
