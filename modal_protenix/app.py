"""
Protenix Modal Inference Endpoint

POST /predict      Ab-Ag structure prediction → CIF + confidence + PAE matrix
POST /compute_msa  MMseqs2 MSA search for a sequence, cached by MD5
GET  /health       Health check
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import modal
from pydantic import BaseModel, Field

APP_NAME = "protenix-inference"
DEFAULT_MODEL = "protenix_base_default_v1.0.0"
WEIGHTS_PATH = "/vol/weights"
MSA_PATH = "/vol/msa"
PROTENIX_SRC = "/protenix"

weights_vol = modal.Volume.from_name("protenix-weights", create_if_missing=True)
msa_vol = modal.Volume.from_name("protenix-msa-cache", create_if_missing=True)

TORCH_EXT_CACHE = "/vol/weights/torch_ext"

# nvidia/cuda devel image provides nvcc + CUDA headers and sets CUDA_HOME=/usr/local/cuda,
# which is required for JIT-compiling Protenix's fast_layernorm and cuequivariance kernels.
# debian_slim only ships the driver API (libcuda.so), not the full toolkit.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("git", "g++", "gcc", "libc6-dev", "make", "hmmer", "kalign")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .add_local_dir(
        local_path="/home/aleph/Desktop/agentic/Protenix",
        remote_path=PROTENIX_SRC,
        copy=True,
    )
    .run_commands(
        f"grep -v '^torch' {PROTENIX_SRC}/requirements.txt > /tmp/req_notorch.txt"
        f" && pip install -r /tmp/req_notorch.txt",
        f"pip install -e {PROTENIX_SRC}",
        "pip install 'fastapi[standard]' httpx",
    )
    .env({
        "PROTENIX_ROOT_DIR": WEIGHTS_PATH,
        "LAYERNORM_TYPE": "fast_layernorm",
        # Cache JIT-compiled kernels to the weights volume so they survive container
        # restarts. First cold start after deployment compiles once; all subsequent
        # starts load the cached .so directly (~seconds vs ~minutes).
        "TORCH_EXTENSIONS_DIR": TORCH_EXT_CACHE,
    })
)


def _download_weights():
    """Download checkpoint and CCD data into the weights Volume at build time."""
    import os
    import sys
    import requests

    sys.path.insert(0, PROTENIX_SRC)
    os.environ["PROTENIX_ROOT_DIR"] = WEIGHTS_PATH

    from protenix.web_service.dependency_url import URL

    def _fetch(url: str, dest: str) -> None:
        if os.path.exists(dest):
            print(f"  Cached ({os.path.getsize(dest) // 1_000_000} MB): {dest}")
            return
        print(f"  Downloading {url}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        r = requests.get(url, stream=True, timeout=600)
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(8192):
                fh.write(chunk)
        print(f"  Saved ({os.path.getsize(dest) // 1_000_000} MB): {dest}")

    for key, fname in [
        ("ccd_components_file", "components.cif"),
        ("ccd_components_rdkit_mol_file", "components.cif.rdkit_mol.pkl"),
        ("pdb_cluster_file", "clusters-by-entity-40.txt"),
        ("obsolete_release_data_csv", "obsolete_release_date.csv"),
    ]:
        _fetch(URL[key], f"{WEIGHTS_PATH}/common/{fname}")

    _fetch(URL[DEFAULT_MODEL], f"{WEIGHTS_PATH}/checkpoint/{DEFAULT_MODEL}.pt")

    weights_vol.commit()
    print("Weights volume ready.")


image = image.run_function(
    _download_weights,
    volumes={WEIGHTS_PATH: weights_vol},
    timeout=1800,
)

app = modal.App(APP_NAME, image=image)


class PredictRequest(BaseModel):
    antigen_seq: str
    heavy_chain_seq: str
    light_chain_seq: str
    antigen_msa_key: Optional[str] = Field(
        None,
        description="MD5 key from POST /compute_msa. Injects cached antigen MSA; "
                    "heavy/light get single-seq stubs so online search is skipped.",
    )
    model_name: str = DEFAULT_MODEL
    seeds: List[int] = Field(
        [101],
        description="Fixed seeds → deterministic PAE/ipSAE. "
                    "Protenix calls seed_everything(); Boltz-2 has no seed API.",
    )
    n_sample: int = Field(1, ge=1, le=5)
    dtype: str = "bf16"
    use_tfg_guidance: bool = False
    use_template: bool = False
    epitopes: Optional[dict] = Field(
        None,
        description='{"positions": [int, ...], "max_distance": float}',
    )


class MSARequest(BaseModel):
    sequence: str


class SampleResult(BaseModel):
    structure_cif: str
    confidence: dict  # plddt, ptm, iptm, chain_pair_iptm, ranking_score, ...
    pae: list         # token_pair_pae [N_token×N_token]; chain order: antigen, heavy, light


class PredictResponse(BaseModel):
    samples:    List[SampleResult]
    n_samples:  int
    model_name: str
    elapsed_s:  float


@app.cls(
    gpu=["A100-40GB", "A100-80GB"],
    volumes={WEIGHTS_PATH: weights_vol, MSA_PATH: msa_vol},
    timeout=900,
    image=image,
)
@modal.concurrent(max_inputs=1)
class ProtenixRunner:

    @modal.enter()
    def setup(self) -> None:
        import sys
        sys.path.insert(0, PROTENIX_SRC)
        os.environ["PROTENIX_ROOT_DIR"] = WEIGHTS_PATH
        os.environ["LAYERNORM_TYPE"] = "fast_layernorm"

        ckpt = Path(f"{WEIGHTS_PATH}/checkpoint/{DEFAULT_MODEL}.pt")
        if not ckpt.exists():
            raise RuntimeError(
                f"Checkpoint not found at {ckpt}. "
                "Run `modal deploy modal_protenix/app.py` first."
            )
        print(f"[ProtenixRunner] ready — {ckpt} ({ckpt.stat().st_size // 1_000_000} MB)")

        # Trigger fast_layernorm JIT compilation now (at container startup) so the
        # first predict call isn't penalised. The compiled .so is written to
        # TORCH_EXTENSIONS_DIR which lives on the weights volume; commit so it
        # survives container restarts and doesn't need recompiling next cold start.
        ext_so = Path(TORCH_EXT_CACHE).rglob("fast_layer_norm_cuda_v2*.so")
        if not any(True for _ in ext_so):
            print("[setup] Compiling fast_layernorm kernel and caching to volume…")
            from protenix.model.layer_norm.layer_norm import FusedLayerNorm  # noqa: F401
            weights_vol.commit()
            print("[setup] fast_layernorm cached.")

    def _build_input_json(self, req: dict, job_name: str, msa_dir: Optional[Path], tmp: Path) -> list:
        """
        Build the protenix pred input JSON.
        Entity order (1-indexed): 1=antigen, 2=heavy, 3=light.

        When antigen MSA is cached, antigen gets real .a3m paths and heavy/light get
        single-seq stubs so update_infer_json skips the online search for all chains.
        """
        antigen_chain: dict = {"sequence": req["antigen_seq"], "count": 1}
        heavy_chain: dict = {"sequence": req["heavy_chain_seq"], "count": 1}
        light_chain: dict = {"sequence": req["light_chain_seq"], "count": 1}

        has_real_msa = False
        if msa_dir is not None and msa_dir.exists():
            paired = msa_dir / "pairing.a3m"
            unpaired = msa_dir / "non_pairing.a3m"
            if paired.exists():
                antigen_chain["pairedMsaPath"] = str(paired)
                has_real_msa = True
            if unpaired.exists():
                antigen_chain["unpairedMsaPath"] = str(unpaired)

        if has_real_msa:
            for chain_dict, name, seq in [
                (heavy_chain, "heavy", req["heavy_chain_seq"]),
                (light_chain, "light", req["light_chain_seq"]),
            ]:
                stub = tmp / f"{name}_stub.a3m"
                stub.write_text(f">query\n{seq}\n")
                chain_dict["pairedMsaPath"] = str(stub)
                chain_dict["unpairedMsaPath"] = str(stub)

        sample: dict = {
            "name": job_name,
            "sequences": [
                {"proteinChain": antigen_chain},
                {"proteinChain": heavy_chain},
                {"proteinChain": light_chain},
            ],
            "covalent_bonds": [],
        }

        # entity 1 = antigen pocket surface, entity 2 = heavy binder
        epitopes = req.get("epitopes")
        if epitopes and epitopes.get("positions"):
            contact_residues = [
                {"entity": 1, "copy": 1, "position": int(p)}
                for p in epitopes["positions"]
            ]
            sample["constraint"] = {
                "pocket": {
                    "binder_chain": {"entity": 2, "copy": 1},
                    "contact_residues": contact_residues,
                    "max_distance": float(epitopes.get("max_distance", 6.0)),
                }
            }

        return [sample]

    @modal.method()
    def predict(self, payload: dict) -> dict:
        t0 = time.time()

        msa_key = payload.get("antigen_msa_key")
        msa_dir = Path(f"{MSA_PATH}/{msa_key}") if msa_key else None
        job_name = f"job_{int(time.time() * 1000)}"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_json = tmp / "input.json"
            dump_dir = tmp / "output"
            dump_dir.mkdir()

            body = self._build_input_json(payload, job_name, msa_dir, tmp)
            input_json.write_text(json.dumps(body, indent=2))

            # use_msa=true when MSA paths are injected; update_infer_json skips online search
            has_msa = msa_dir is not None and (msa_dir / "pairing.a3m").exists()
            use_msa_flag = "true" if has_msa else "false"

            seeds_str = ",".join(str(s) for s in payload.get("seeds", [101]))
            model_name = payload.get("model_name", DEFAULT_MODEL)

            cmd = [
                "protenix", "pred",
                "-i", str(input_json),
                "-o", str(dump_dir),
                "--model_name",          model_name,
                "--seeds",               seeds_str,
                "--sample",              str(payload.get("n_sample", 1)),
                "--dtype",               payload.get("dtype", "bf16"),
                "--use_default_params",  "true",
                "--need_atom_confidence","true",   # emit full_data with token_pair_pae
                "--use_msa",             use_msa_flag,
                "--trimul_kernel",       "torch",
                "--triatt_kernel",       "torch",
                "--enable_cache",        "true",
                "--enable_fusion",       "true",
                "--enable_tf32",         "true",
                "--use_template",        str(payload.get("use_template", False)).lower(),
                "--use_rna_msa",         "false",
                "--use_seeds_in_json",   "false",
            ]

            if payload.get("use_tfg_guidance"):
                cmd += ["--use_tfg_guidance", "true"]

            print(f"[predict] {' '.join(cmd)}")
            env = {**os.environ, "PROTENIX_ROOT_DIR": WEIGHTS_PATH}
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)

            # Always surface Protenix logs — inference_jsons catches all exceptions
            # internally and exits 0 with just a WARNING in stderr on failure.
            if result.stdout:
                print(f"[protenix stdout]\n{result.stdout[-6000:]}")
            if result.stderr:
                print(f"[protenix stderr]\n{result.stderr[-6000:]}")

            if result.returncode != 0:
                raise RuntimeError(
                    f"protenix pred exited {result.returncode}\n"
                    f"STDOUT:\n{result.stdout[-4000:]}\n"
                    f"STDERR:\n{result.stderr[-4000:]}"
                )

            return self._parse_outputs(
                dump_dir, model_name, time.time() - t0,
                stdout=result.stdout, stderr=result.stderr,
            )

    def _parse_outputs(
        self, dump_dir: Path, model_name: str, elapsed: float,
        stdout: str = "", stderr: str = "",
    ) -> dict:
        log_tail = f"\nSTDOUT:\n{stdout[-3000:]}\nSTDERR:\n{stderr[-3000:]}"

        cif_files  = sorted(dump_dir.rglob("*_sample_*.cif"))
        conf_files = sorted(dump_dir.rglob("*_summary_confidence_sample_*.json"))
        full_files = sorted(dump_dir.rglob("*_full_data_sample_*.json"))

        if not cif_files:
            raise RuntimeError(f"No CIF output under {dump_dir}.{log_tail}")
        if not conf_files:
            raise RuntimeError(f"No summary_confidence JSON under {dump_dir}.{log_tail}")
        if not full_files:
            raise RuntimeError(f"No full_data JSON under {dump_dir}.{log_tail}")

        samples = []
        for cif_f, conf_f, full_f in zip(cif_files, conf_files, full_files):
            full_data = json.loads(full_f.read_text())
            raw_pae = full_data.get("token_pair_pae", [])
            if hasattr(raw_pae, "tolist"):
                raw_pae = raw_pae.tolist()
            samples.append({
                "structure_cif": cif_f.read_text(),
                "confidence":    json.loads(conf_f.read_text()),
                "pae":           raw_pae,
            })

        return {
            "samples":    samples,
            "n_samples":  len(samples),
            "model_name": model_name,
            "elapsed_s":  round(elapsed, 2),
        }

    @modal.method()
    def compute_msa(self, sequence: str) -> str:
        """
        Run MMseqs2 MSA search and cache results under MSA_PATH/{MD5}.
        Returns the MD5 key; returns immediately on cache hit.
        """
        import sys
        sys.path.insert(0, PROTENIX_SRC)
        os.environ["PROTENIX_ROOT_DIR"] = WEIGHTS_PATH

        key = hashlib.md5(sequence.encode()).hexdigest()
        dest = Path(f"{MSA_PATH}/{key}")

        if (dest / "pairing.a3m").exists() and (dest / "non_pairing.a3m").exists():
            print(f"[compute_msa] cache hit: {key}")
            return key

        dest.mkdir(parents=True, exist_ok=True)
        print(f"[compute_msa] searching len={len(sequence)}, key={key}")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                from runner.msa_search import msa_search
                result_dirs = msa_search([sequence], tmpdir, mode="protenix")
                if not result_dirs:
                    raise RuntimeError("msa_search returned no result directories")
                src_dir = Path(result_dirs[0])
                for fname in ["pairing.a3m", "non_pairing.a3m"]:
                    src = src_dir / fname
                    if src.exists():
                        shutil.copy(src, dest / fname)
                    else:
                        print(f"[compute_msa] warning: {fname} not in {src_dir}")
        except Exception as exc:
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError(f"MSA search failed: {exc}") from exc

        msa_vol.commit()
        print(f"[compute_msa] cached → key={key}")
        return key


@app.function(image=image)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, HTTPException

    web_app = FastAPI(title="Protenix Inference API", version="1.0.0")
    runner = ProtenixRunner()

    @web_app.post("/predict", response_model=PredictResponse)
    async def predict(body: PredictRequest):
        try:
            return await runner.predict.remote.aio(body.model_dump())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @web_app.post("/compute_msa")
    async def compute_msa(body: MSARequest):
        try:
            key = await runner.compute_msa.remote.aio(body.sequence)
            return {"msa_cache_key": key, "sequence_length": len(body.sequence)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": DEFAULT_MODEL, "app": APP_NAME}

    return web_app
