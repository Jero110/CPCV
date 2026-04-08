import os
import pickle
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"


def _fname(kind: str, n_perm: int, is_frac: float, seed: int) -> Path:
    tag = f"mc_{kind}_N{n_perm}_IS{int(is_frac*100):03d}_seed{seed}.pkl"
    return CACHE_DIR / tag


def load(kind: str, n_perm: int, is_frac: float, seed: int):
    """Returns cached results or None if not found."""
    path = _fname(kind, n_perm, is_frac, seed)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save(kind: str, n_perm: int, is_frac: float, seed: int, data):
    CACHE_DIR.mkdir(exist_ok=True)
    path = _fname(kind, n_perm, is_frac, seed)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[cache] saved → {path.name}")


def list_cached():
    files = sorted(CACHE_DIR.glob("mc_*.pkl"))
    if not files:
        print("[cache] empty")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.0f} KB)")
