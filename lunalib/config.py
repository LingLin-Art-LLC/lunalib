"""Runtime configuration profiles for LunaLib."""
from __future__ import annotations

import os
from typing import Dict

PROFILE = os.getenv("LUNALIB_PROFILE", "fast")

PROFILES: Dict[str, Dict[str, str]] = {
    "fast": {
        "LUNALIB_SM2_BACKEND": "phos",
        "LUNALIB_SM2_GPU": "1",
        "LUNALIB_CUDA_BATCH_SIZE": "16000000",
        "LUNALIB_CUDA_CHUNK_SIZE": "500000",
        "LUNALIB_CUDA_THREADS": "256",
        "LUNALIB_CUDA_BLOCKS": "65536",
        "LUNALIB_CUDA_ITERS": "64",
        "LUNALIB_MINING_HASH_MODE": "compact",
        "LUNALIB_WALLET_SYNC_LOOKBACK": "200",
        "LUNALIB_WALLET_CIPHER": "sm4",
        "LUNALIB_SM4_USE_GPU": "1",
        "LUNALIB_SM4_CUDA_KERNEL": "1",
        "LUNALIB_SM4_CTR_CHUNK_BLOCKS": "131072",
        "LUNALIB_SM4_CTR_GPU_XOR": "1",
        "LUNALIB_SM4_MIN_BLOCKS": "8",
        "LUNALIB_SM4_XOR_MIN_BYTES": "4096",
        "LUNALIB_SM4_GPU_CHUNK_BYTES": "0",
    }
}


def apply_profile() -> None:
    profile = os.getenv("LUNALIB_PROFILE", PROFILE)
    if not profile:
        return
    settings = PROFILES.get(profile)
    if not settings:
        return
    for key, value in settings.items():
        os.environ.setdefault(key, value)
