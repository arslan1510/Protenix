"""Quick smoke test for the Modal Protenix endpoint.

Usage:
    python modal_protenix/test.py https://<workspace>--protenix-inference-api.modal.run
"""
import sys
import time
from pathlib import Path

import httpx

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"

# 5M2J — antigen chain A (139 aa from ATOM records)
ANTIGEN_SEQ = (
    "SDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLT"
    "HTISRIAVSYQTKVNLLSAIKSPCQPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL"
)
HEAVY_SEQ = (
    "EVQLLESGGGLVQPGGSLRLSCAASGFDLAAGAMSWVRQAPGKGLEWVSAISPSGGSTYYADSVK"
    "GRFTISRDNSKNTLYLQMNSLRAEDTAVYYFGYENRGYGELDFWGQGTLV"
)
LIGHT_SEQ = (
    "DIQMTQSPSSLSASVGDRVTITSKASAPVDGKAGWYQQKPGKAPKLLIYGSTERRGGVPSRFSGS"
    "GSGTDFTLTISSLQPEDFATYYFMDSSGGSKGFGQGTKVEIKRTV"
)

PAYLOAD = {
    "antigen_seq":    ANTIGEN_SEQ,
    "heavy_chain_seq": HEAVY_SEQ,
    "light_chain_seq": LIGHT_SEQ,
    "epitopes": {
        # positions are 1-indexed in the antigen sequence (139 aa from ATOM records)
        # drop 141, 145, 146 — out of range for this chain
        "positions": [19,20,21,23,65,67,71,77,79,81,89,91,135,137,138],
        "max_distance": 6.0,
    },
    "seeds": [101],
    "n_sample": 5,
    "dtype": "bf16",
}


def test_health():
    r = httpx.get(f"{BASE_URL}/health", timeout=10, follow_redirects=True)
    r.raise_for_status()
    print("health:", r.json())


def compute_msa(antigen_seq: str) -> str:
    """Call /compute_msa for the antigen and return the cache key."""
    print(f"\nPOST /compute_msa  (antigen={len(antigen_seq)}aa) — may take ~2 min…")
    t0 = time.time()
    r = httpx.post(
        f"{BASE_URL}/compute_msa",
        json={"sequence": antigen_seq},
        timeout=300,
        follow_redirects=True,
    )
    if r.status_code != 200:
        print(f"  MSA ERROR {r.status_code}: {r.text}")
        return None
    key = r.json().get("msa_cache_key")
    print(f"  msa_cache_key: {key}  ({time.time()-t0:.1f}s)")
    return key


def test_predict(antigen_msa_key: str = None):
    payload = dict(PAYLOAD)
    if antigen_msa_key:
        payload["antigen_msa_key"] = antigen_msa_key

    print(f"\nPOST /predict  (antigen={len(payload['antigen_seq'])}aa, "
          f"heavy={len(payload['heavy_chain_seq'])}aa, "
          f"light={len(payload['light_chain_seq'])}aa, "
          f"msa={'yes' if antigen_msa_key else 'no'})")
    print("Waiting for GPU container + inference (~2-5 min on cold start)…")

    t0 = time.time()
    r = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=600, follow_redirects=True)
    elapsed = time.time() - t0

    if r.status_code != 200:
        print(f"ERROR {r.status_code}:\n{r.text}")
        sys.exit(1)

    result = r.json()
    samples = result.get("samples", [])

    print(f"\nResult ({elapsed:.1f}s wall, {result.get('elapsed_s')}s model):")
    print(f"  n_samples : {result.get('n_samples')}")
    for i, s in enumerate(samples):
        conf = s.get("confidence", {})
        pae  = s.get("pae", [])
        print(f"  sample {i}: ranking={conf.get('ranking_score', 'n/a'):.4f}  "
              f"iptm={conf.get('iptm', 'n/a'):.4f}  "
              f"pae={len(pae)}x{len(pae[0]) if pae else 0}")

    import json as _json
    base = Path(__file__).parent / "test_output"
    base.mkdir(exist_ok=True)
    tag = "msa" if payload.get("antigen_msa_key") else "nomsa"
    # auto-increment run directory: run1_msa, run2_msa, ...
    run_idx = 1
    while (base / f"run{run_idx}_{tag}").exists():
        run_idx += 1
    out_dir = base / f"run{run_idx}_{tag}"
    out_dir.mkdir()
    for i, s in enumerate(samples):
        (out_dir / f"structure_sample{i}.cif").write_text(s.get("structure_cif", ""))
        (out_dir / f"confidence_sample{i}.json").write_text(_json.dumps(s.get("confidence", {}), indent=2))
        (out_dir / f"pae_sample{i}.json").write_text(_json.dumps(s.get("pae", []), indent=2))
    (out_dir / "meta.json").write_text(_json.dumps(
        {k: v for k, v in result.items() if k != "samples"}, indent=2,
    ))
    print(f"\n  Saved {len(samples)} samples to {out_dir}/")

    # if a previous run exists, diff ranking_scores to check determinism
    prev_dir = base / f"run{run_idx - 1}_{tag}" if run_idx > 1 else None
    if prev_dir and prev_dir.exists():
        print("\n  Determinism check vs previous run:")
        match = True
        for i in range(len(samples)):
            cur  = _json.loads((out_dir  / f"confidence_sample{i}.json").read_text())
            prev = _json.loads((prev_dir / f"confidence_sample{i}.json").read_text())
            cur_r, prev_r = cur.get("ranking_score"), prev.get("ranking_score")
            same = abs(cur_r - prev_r) < 1e-9 if (cur_r and prev_r) else cur_r == prev_r
            flag = "✓" if same else "✗ DIFF"
            print(f"    sample {i}: ranking {prev_r:.10f} → {cur_r:.10f}  {flag}")
            if not same:
                match = False
        print(f"  {'All outputs identical ✓' if match else 'DIFFERENCES FOUND ✗'}")


if __name__ == "__main__":
    test_health()
    msa_key = compute_msa(ANTIGEN_SEQ)
    test_predict(antigen_msa_key=msa_key)
