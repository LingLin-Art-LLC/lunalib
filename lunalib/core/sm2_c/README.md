# SM2 CPython Backend (C Extension)

This folder contains a CPython C-extension scaffold for an SM2 signing backend.

## Status
- Mod-p math and affine point ops are implemented in C.
- Signature still uses Python for hash and mod-n math (not fully optimized yet).

## Next Steps
1. Implement 256-bit big integer math (add/sub/mul/mod/inv).
2. Implement SM2 curve point ops (add/double/scalar multiply).
3. Implement SM2 signing (`r`, `s`) using random `k`.

## Build
From repo root:
```
python -m pip install -e .
```

Then test in Python:
```
from lunalib.core.sm2_c import sm2_ext
```
