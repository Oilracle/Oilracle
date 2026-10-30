# Vendored Kronos model code

Source: https://github.com/shiyu-coder/Kronos, branch `master`, commit
`67b630e67f6a18c9e9be918d9b4337c960db1e9a`, downloaded 2026-10-17. MIT License (see `LICENSE`).

| File | Original SHA-256 |
| ---- | ---------------- |
| `model/kronos.py` → `kronos.py` | `0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032` |
| `model/module.py` → `module.py` | `a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f` |
| `LICENSE` | `acb2d194d378204e5f2be4dcd24d39ecac437903620c790c3315a96dab388fdc` |

**Only modification:** in `kronos.py`, the lines `import sys`, `sys.path.append("../")` and
`from model.module import *` were replaced by the package-relative `from .module import *`.
`module.py` is unchanged.

Only the model classes (`KronosTokenizer`, `Kronos`) and their modules are used. Training and
generation are our own code in `m3_ml/kronos/` (the original generation averages all sampled paths
and returns only the mean; see `m3_ml/kronos/PLAN.md`).
