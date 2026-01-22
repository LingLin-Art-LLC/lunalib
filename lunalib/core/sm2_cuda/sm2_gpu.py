import os
from typing import Iterable, List

from lunalib.core.sm2 import SM2

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except Exception:
    cp = None
    CUDA_AVAILABLE = False

BASE_DIR = os.path.dirname(__file__)
KERNEL_PATH = os.path.join(BASE_DIR, "sm2_kernel.cu")
CURVE_PATH = os.path.join(BASE_DIR, "sm2_curve_params.h")
FIELD_PATH = os.path.join(BASE_DIR, "sm2_field.cu")


def _load_kernel(name: str) -> "cp.RawKernel":
    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available")
    with open(CURVE_PATH, "r", encoding="utf-8") as f:
        curve_src = f.read()
    with open(FIELD_PATH, "r", encoding="utf-8") as f:
        field_src = f.read()
    with open(KERNEL_PATH, "r", encoding="utf-8") as f:
        kernel_src = f.read()
    field_lines = []
    for line in field_src.splitlines():
        if "sm2_curve_params.h" in line:
            continue
        field_lines.append(line)
    field_src = "\n".join(field_lines)
    lines = []
    for line in kernel_src.splitlines():
        if "sm2_curve_params.h" in line or "sm2_field.cu" in line:
            continue
        lines.append(line)
    kernel_src = "\n".join(lines)
    src = "\n".join(["// SM2_KERNEL_BUILD=3", curve_src, field_src, kernel_src])
    return cp.RawKernel(src, name)


def sm2_sign_gpu(message: bytes, private_keys: Iterable[bytes], use_gpu: bool | None = None) -> List[bytes]:
    """SM2 signing with optional GPU assist for k*G.

    Returns list of signatures as raw bytes (r||s).
    """
    if use_gpu is None:
        use_gpu = os.getenv("LUNALIB_SM2_GPU", "0") == "1"

    sm2 = SM2()
    priv_list = list(private_keys)

    if not use_gpu:
        signatures = []
        for priv in priv_list:
            sig_hex = sm2.sign(message, priv.hex())
            signatures.append(bytes.fromhex(sig_hex))
        return signatures

    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available for SM2 GPU signing")

    # Compute e = H(Z || message)
    e_bytes = sm2.hash.hash(sm2.Z + message)
    e = int.from_bytes(e_bytes, "big")

    # Generate random nonces k on CPU (cryptographically secure)
    import secrets
    k_list = []
    for _ in priv_list:
        k = secrets.randbelow(sm2.curve.n - 1) + 1
        k_list.append(k)

    # Run GPU kernel to compute k*G and get x1
    kernel = _load_kernel("sm2_sign_kernel")
    batch = len(priv_list)
    nonces = bytearray()
    for k in k_list:
        nonces += k.to_bytes(32, "big")

    nonces_gpu = cp.asarray(nonces, dtype=cp.uint8)
    pub_out_gpu = cp.empty(batch * 64, dtype=cp.uint8)

    threads = 128
    blocks = (batch + threads - 1) // threads
    kernel((blocks,), (threads,), (cp.asarray(bytearray(), dtype=cp.uint8), 0, cp.asarray(bytearray(), dtype=cp.uint8), 0,
                                   nonces_gpu, 32, pub_out_gpu, 64, batch))

    pub_out = cp.asnumpy(pub_out_gpu).tobytes()

    signatures = []
    for i, priv in enumerate(priv_list):
        d = int.from_bytes(priv, "big")
        x1 = int.from_bytes(pub_out[i * 64:i * 64 + 32], "big")

        r = (e + x1) % sm2.curve.n
        if r == 0 or r + k_list[i] == sm2.curve.n:
            raise ValueError("Invalid r encountered during SM2 signing")

        d_plus_1_inv = sm2.curve.mod_inv(1 + d, sm2.curve.n)
        s = (d_plus_1_inv * (k_list[i] - r * d)) % sm2.curve.n
        if s == 0:
            raise ValueError("Invalid s encountered during SM2 signing")

        sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        signatures.append(sig)

    return signatures


def sm2_verify_gpu(message: bytes, public_keys: Iterable[bytes], signatures: Iterable[bytes], use_gpu: bool | None = None) -> List[bool]:
    """SM2 verification with optional GPU path (currently CPU fallback)."""
    if use_gpu is None:
        use_gpu = os.getenv("LUNALIB_SM2_GPU", "0") == "1"

    sm2 = SM2()
    pub_list = list(public_keys)
    sig_list = list(signatures)

    if not use_gpu:
        results = []
        for pub, sig in zip(pub_list, sig_list):
            sig_hex = sig.hex()
            pub_hex = pub.hex()
            results.append(sm2.verify(message, sig_hex, pub_hex))
        return results

    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available for SM2 GPU verification")

    # Compute e = H(Z || message)
    e_bytes = sm2.hash.hash(sm2.Z + message)
    e = int.from_bytes(e_bytes, "big")

    # Prepare scalars s and t for each signature
    scalars = bytearray()
    pub_bytes = bytearray()
    batch = len(pub_list)
    for pub, sig in zip(pub_list, sig_list):
        if len(sig) != 64:
            raise ValueError("Signature must be 64 bytes (r||s)")
        r = int.from_bytes(sig[:32], "big")
        s = int.from_bytes(sig[32:], "big")
        t = (r + s) % sm2.curve.n
        scalars += s.to_bytes(32, "big") + t.to_bytes(32, "big")

        # Accept public key with optional 0x04 prefix
        if len(pub) == 65 and pub[0] == 0x04:
            pub = pub[1:]
        if len(pub) != 64:
            raise ValueError("Public key must be 64 bytes (x||y) or 65 bytes with 0x04 prefix")
        pub_bytes += pub

    kernel = _load_kernel("sm2_verify_x1_kernel")
    scalars_gpu = cp.asarray(scalars, dtype=cp.uint8)
    pubs_gpu = cp.asarray(pub_bytes, dtype=cp.uint8)
    x_out_gpu = cp.empty(batch * 32, dtype=cp.uint8)

    threads = 128
    blocks = (batch + threads - 1) // threads
    kernel((blocks,), (threads,), (pubs_gpu, 64, scalars_gpu, 64, x_out_gpu, 32, batch))

    x_out = cp.asnumpy(x_out_gpu).tobytes()

    results = []
    for i, sig in enumerate(sig_list):
        r = int.from_bytes(sig[:32], "big")
        x1 = int.from_bytes(x_out[i * 32:(i + 1) * 32], "big")
        R = (e + x1) % sm2.curve.n
        results.append(R == r)

    return results


def sm2_keygen_gpu(entropy_inputs: Iterable[bytes], use_gpu: bool | None = None) -> List[bytes]:
    """SM2 key generation with optional GPU path (currently CPU fallback).

    Returns private keys as list of 32-byte values.
    """
    if use_gpu is None:
        use_gpu = os.getenv("LUNALIB_SM2_GPU", "0") == "1"

    sm2 = SM2()
    ent_list = list(entropy_inputs)

    if not use_gpu:
        priv_keys = []
        for _ in ent_list:
            priv_hex, _ = sm2.generate_keypair()
            priv_keys.append(bytes.fromhex(priv_hex))
        return priv_keys

    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available for SM2 GPU keygen")

    # GPU uses provided entropy as private key bytes directly
    for ent in ent_list:
        if len(ent) != 32:
            raise ValueError("Entropy input must be 32 bytes for SM2 keygen")

    kernel = _load_kernel("sm2_keygen_kernel")
    batch = len(ent_list)
    ent_bytes = bytearray().join(ent_list)

    ent_gpu = cp.asarray(ent_bytes, dtype=cp.uint8)
    priv_out_gpu = cp.empty(batch * 32, dtype=cp.uint8)
    pub_out_gpu = cp.empty(batch * 64, dtype=cp.uint8)

    threads = 128
    blocks = (batch + threads - 1) // threads
    kernel((blocks,), (threads,), (ent_gpu, 32, priv_out_gpu, 32, pub_out_gpu, 64, batch))

    priv_out = cp.asnumpy(priv_out_gpu).tobytes()
    pub_out = cp.asnumpy(pub_out_gpu).tobytes()

    # Return private keys; public keys are available via sm2_keygen_gpu_with_pub
    priv_keys = [priv_out[i * 32:(i + 1) * 32] for i in range(batch)]
    return priv_keys


def sm2_keygen_gpu_with_pub(entropy_inputs: Iterable[bytes], use_gpu: bool | None = None) -> List[tuple[bytes, bytes]]:
    """SM2 key generation that returns (priv, pub) tuples.

    Public key is 64-byte (x||y), no 0x04 prefix.
    """
    if use_gpu is None:
        use_gpu = os.getenv("LUNALIB_SM2_GPU", "0") == "1"

    sm2 = SM2()
    ent_list = list(entropy_inputs)

    if not use_gpu:
        out = []
        for _ in ent_list:
            priv_hex, pub_hex = sm2.generate_keypair()
            pub_bytes = bytes.fromhex(pub_hex[2:]) if pub_hex.startswith("04") else bytes.fromhex(pub_hex)
            out.append((bytes.fromhex(priv_hex), pub_bytes))
        return out

    if not CUDA_AVAILABLE:
        raise RuntimeError("CuPy is not available for SM2 GPU keygen")

    for ent in ent_list:
        if len(ent) != 32:
            raise ValueError("Entropy input must be 32 bytes for SM2 keygen")

    kernel = _load_kernel("sm2_keygen_kernel")
    batch = len(ent_list)
    ent_bytes = bytearray().join(ent_list)

    ent_gpu = cp.asarray(ent_bytes, dtype=cp.uint8)
    priv_out_gpu = cp.empty(batch * 32, dtype=cp.uint8)
    pub_out_gpu = cp.empty(batch * 64, dtype=cp.uint8)

    threads = 128
    blocks = (batch + threads - 1) // threads
    kernel((blocks,), (threads,), (ent_gpu, 32, priv_out_gpu, 32, pub_out_gpu, 64, batch))

    priv_out = cp.asnumpy(priv_out_gpu).tobytes()
    pub_out = cp.asnumpy(pub_out_gpu).tobytes()

    result = []
    for i in range(batch):
        priv = priv_out[i * 32:(i + 1) * 32]
        pub = pub_out[i * 64:(i + 1) * 64]
        result.append((priv, pub))
    return result
