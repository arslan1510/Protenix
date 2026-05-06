"""
RunPod Serverless Handler for Protenix Inference.

Based on the working dropxcell-proteinx/api_server/inference_engine.py pattern.
"""
import logging
import os
import shutil
import tempfile
import json
import runpod
import torch
import traceback
import platform
from argparse import Namespace
from typing import Any, Dict, List, Union

# Configure logging first
LOG_FORMAT = "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import Protenix modules
from ml_collections.config_dict import ConfigDict
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config import parse_configs
from runner.inference import (
    InferenceRunner,
    infer_predict,
    download_infercence_cache,
)

# PyTorch 2.6+ compatibility fix for ESM model loading
# The fair-esm repository is archived and can't be updated to support newer PyTorch versions.
# ESM model files contain argparse.Namespace which isn't allowed by default in secure unpickling.
torch.serialization.add_safe_globals([Namespace])

# Global state (initialized once on cold start)
cache_downloaded = False

DEFAULT_MODEL_NAME = "protenix-v2"
MODEL_ALIASES = {
    "v2": "protenix-v2",
    "protenix2": "protenix-v2",
    "protenix_v2": "protenix-v2",
}
MODEL_SAMPLE_KEYS = {
    "name",
    "sequences",
    "covalent_bonds",
}
OPTIONAL_CONFIG_KEYS = {
    "dtype",
    "need_atom_confidence",
    "sorted_by_ranking_score",
    "enable_tf32",
    "enable_efficient_fusion",
    "enable_diffusion_shared_vars_cache",
    "use_template",
    "use_rna_msa",
    "use_seeds_in_json",
    "msa_pair_as_unpair",
}


def log_separator(title: str = ""):
    """Log a visual separator for clarity."""
    logger.info("=" * 60)
    if title:
        logger.info(title)
        logger.info("=" * 60)


def log_runtime_info():
    """Log runtime environment information for debugging."""
    logger.info("RUNTIME ENVIRONMENT:")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"  CUDA device name: {torch.cuda.get_device_name(0)}")


def resolve_model_name(model_name: Any) -> str:
    """Normalize public model aliases to the checkpoint names used by Protenix."""
    if model_name is None:
        return DEFAULT_MODEL_NAME
    normalized = str(model_name).strip()
    return MODEL_ALIASES.get(normalized.lower(), normalized)


def default_model_name() -> str:
    """Return the default model, allowing RunPod env config to override v2."""
    return resolve_model_name(os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))


def apply_optional_request_configs(configs: Any, request_params: Dict[str, Any]) -> None:
    """Apply optional RunPod request params that map directly to Protenix configs."""
    for key in OPTIONAL_CONFIG_KEYS:
        if key in request_params:
            setattr(configs, key, request_params[key])

    if "use_tfg_guidance" not in request_params:
        return

    guidance = getattr(configs.sample_diffusion, "guidance", None)
    if guidance is None:
        logger.warning(
            "use_tfg_guidance was requested, but this checkout has no TFG config"
        )
        return
    guidance.enable = bool(request_params["use_tfg_guidance"])


def init_handler():
    """
    Initialize handler - download checkpoints/cache once on cold start.
    
    This is called once when the container starts.
    """
    global cache_downloaded
    
    log_separator("INIT_HANDLER STARTING")
    log_runtime_info()
    
    # Create a minimal config just to download the cache
    logger.info("Creating minimal config for cache download...")
    
    local_inference_configs = dict(inference_configs)
    local_inference_configs["dump_dir"] = "/tmp/init_output"
    local_inference_configs["input_json_path"] = "/dev/null"
    local_inference_configs["model_name"] = default_model_name()
    
    configs = {**configs_base, **{"data": data_configs}, **local_inference_configs}
    configs = parse_configs(
        configs=configs,
        fill_required_with_null=True,
    )
    
    model_name = configs.model_name
    logger.info(f"Model name: {model_name}")
    
    # Update with model-specific configs
    if model_name in model_configs:
        model_specific_configs = ConfigDict(model_configs[model_name])
        configs.update(model_specific_configs)
        logger.info(f"Applied model-specific configs for: {model_name}")
    
    # Download checkpoints/cache if needed (do this once on cold start)
    logger.info("Downloading inference cache if needed...")
    download_infercence_cache(configs)
    cache_downloaded = True
    
    log_separator("INIT_HANDLER COMPLETED")


def _coerce_samples(value: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize a single Protenix sample or batch into a list of samples."""
    if isinstance(value, list):
        samples = value
    elif isinstance(value, dict):
        samples = [value]
    else:
        raise ValueError(
            "Model input must be a Protenix sample object or a list of samples"
        )

    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample at index {idx} must be an object")
        if "sequences" not in sample:
            raise ValueError(f"Sample at index {idx} is missing required key: sequences")
    return samples


def request_to_model_input(job_input: Any, job_id: str) -> List[Dict[str, Any]]:
    """
    Convert a RunPod request payload into the Protenix inference JSON format.

    The handler owns API-only fields such as model_name/use_msa/seeds, so they are
    intentionally stripped before the model-facing JSON is written.
    """
    if isinstance(job_input, list):
        samples = _coerce_samples(job_input)
    elif not isinstance(job_input, dict):
        raise ValueError("RunPod input must be an object or a list of samples")

    elif "model_input" in job_input:
        samples = _coerce_samples(job_input["model_input"])
    elif "samples" in job_input:
        samples = _coerce_samples(job_input["samples"])
    elif "sequences" in job_input:
        samples = _coerce_samples(
            {key: value for key, value in job_input.items() if key in MODEL_SAMPLE_KEYS}
        )
    elif "sequence" in job_input:
        samples = [
            {
                "name": job_input.get("name", f"job_{job_id}"),
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": job_input["sequence"],
                            "count": job_input.get("count", 1),
                        }
                    }
                ],
            }
        ]
    else:
        raise ValueError(
            "RunPod input must include sequences, sequence, samples, or model_input"
        )

    normalized_samples = []
    for idx, sample in enumerate(samples):
        model_sample = {
            key: value for key, value in sample.items() if key in MODEL_SAMPLE_KEYS
        }
        if "name" not in model_sample:
            suffix = "" if len(samples) == 1 else f"_{idx}"
            model_sample["name"] = f"job_{job_id}{suffix}"
        normalized_samples.append(model_sample)

    return normalized_samples


def save_model_input_to_json(samples: List[Dict[str, Any]], output_dir: str) -> str:
    """
    Save model-facing samples to a JSON file in the format expected by Protenix.
    
    Args:
        samples: Protenix model input samples
        output_dir: Directory to save the JSON file
        
    Returns:
        Path to the saved JSON file
    """
    json_path = os.path.join(output_dir, "input.json")
    with open(json_path, "w") as f:
        json.dump(samples, f, indent=2)
    
    logger.info(f"Saved input JSON to: {json_path}")
    return json_path


def collect_output_files(output_dir: str) -> Dict[str, str]:
    """
    Collect all output files from the inference output directory.
    
    Args:
        output_dir: The output directory from inference
        
    Returns:
        Dictionary mapping filename to file contents
    """
    output_files = {}
    
    # Walk through all subdirectories to find output files
    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                # Try to read as text (CIF, JSON files are text)
                with open(filepath, "r") as f:
                    content = f.read()
                # Use relative path from output_dir as key
                rel_path = os.path.relpath(filepath, output_dir)
                output_files[rel_path] = content
                logger.info(f"  Collected: {rel_path} ({len(content)} bytes)")
            except UnicodeDecodeError:
                # Skip binary files
                logger.warning(f"  Skipping binary file: {filepath}")
            except Exception as e:
                logger.error(f"  Error reading file {filepath}: {e}")
    
    logger.info(f"Collected {len(output_files)} output files total")
    return output_files


def model_output_to_response(output_files: Dict[str, str]) -> Dict[str, Any]:
    """
    Convert files emitted by Protenix into a response payload for the RunPod request.

    The raw file map is preserved for existing clients, and common prediction files
    are also exposed as a lightweight structured list.
    """
    predictions = []

    for path in sorted(output_files):
        parts = path.split(os.sep)
        filename = parts[-1]
        prediction = {
            "path": path,
            "filename": filename,
        }

        if "seed_" in path:
            for part in parts:
                if part.startswith("seed_"):
                    prediction["seed"] = part.replace("seed_", "", 1)
                    break

        if filename.endswith(".cif"):
            prediction["type"] = "structure"
        elif filename.endswith(".json"):
            prediction["type"] = "confidence"
        elif path.startswith("ERR/") or f"{os.sep}ERR{os.sep}" in path:
            prediction["type"] = "error"
        else:
            prediction["type"] = "file"

        predictions.append(prediction)

    return {
        "status": "success",
        "output_files": output_files,
        "predictions": predictions,
    }


def handler(event):
    """
    RunPod Handler function.
    
    Uses the proven pattern from dropxcell-proteinx inference_engine.py
    """
    temp_dir = None
    
    try:
        log_separator("HANDLER STARTED")
        
        job_input = event.get("input", {})
        job_id = event.get("id", "unknown")
        
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Event keys: {list(event.keys())}")
        input_keys = list(job_input.keys()) if isinstance(job_input, dict) else []
        logger.info(f"Input keys: {input_keys}")

        # Translate the public request payload into the model-facing Protenix JSON.
        model_samples = request_to_model_input(job_input, job_id)
        request_params = job_input if isinstance(job_input, dict) else {}
        sample_name = ", ".join(sample["name"] for sample in model_samples)
        logger.info(f"Processing sample: {sample_name}")
        
        # Create temporary directory for this job
        temp_dir = tempfile.mkdtemp(prefix=f"protenix_{job_id}_")
        logger.info(f"Created temp directory: {temp_dir}")
        
        # Save model input to JSON file
        json_path = save_model_input_to_json(model_samples, temp_dir)
        
        # Create job-specific output directory
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Extract parameters from request (with defaults)
        model_name = resolve_model_name(
            request_params.get("model_name", default_model_name())
        )
        use_msa = request_params.get("use_msa", True)
        seeds = request_params.get("seeds", [101])
        n_cycle = request_params.get("n_cycle", None)
        n_step = request_params.get("n_step", None)
        n_sample = request_params.get("n_sample", None)
        
        # Ensure seeds is a list
        if not isinstance(seeds, list):
            seeds = [seeds]
        
        logger.info(f"Parameters:")
        logger.info(f"  model_name: {model_name}")
        logger.info(f"  use_msa: {use_msa}")
        logger.info(f"  seeds: {seeds}")
        logger.info(f"  n_cycle: {n_cycle}")
        logger.info(f"  n_step: {n_step}")
        logger.info(f"  n_sample: {n_sample}")
        
        # Set defaults based on model type (matching official Protenix implementation)
        if "mini" in model_name or "tiny" in model_name:
            default_n_cycle = 4
            default_n_step = 5
        else:
            default_n_cycle = 10
            default_n_step = 200
        
        # Handle None, 0, or any falsy value by using defaults
        n_cycle = n_cycle if (n_cycle is not None and n_cycle > 0) else default_n_cycle
        n_step = n_step if (n_step is not None and n_step > 0) else default_n_step
        n_sample = n_sample if (n_sample is not None and n_sample > 0) else 5
        
        logger.info(f"Final parameters after defaults:")
        logger.info(f"  n_cycle: {n_cycle}")
        logger.info(f"  n_step: {n_step}")
        logger.info(f"  n_sample: {n_sample}")
        
        # ============================================================
        # CONFIGURE INFERENCE - Following inference_engine.py pattern
        # ============================================================
        log_separator("CONFIGURING INFERENCE")
        
        # Use copy to avoid global mutation
        local_inference_configs = dict(inference_configs)
        local_inference_configs["dump_dir"] = output_dir
        local_inference_configs["input_json_path"] = json_path
        local_inference_configs["model_name"] = model_name
        
        logger.info("Merging configs...")
        configs = {**configs_base, **{"data": data_configs}, **local_inference_configs}
        
        logger.info("Calling parse_configs()...")
        configs = parse_configs(
            configs=configs,
            fill_required_with_null=True,
        )
        logger.info("parse_configs() completed successfully")
        
        # Update with model-specific configs
        if model_name in model_configs:
            model_specific_configs = ConfigDict(model_configs[model_name])
            configs.update(model_specific_configs)
            logger.info(f"Applied model-specific configs for: {model_name}")
        else:
            logger.warning(f"No model-specific configs found for: {model_name}")
        
        # Set user-provided parameters - PLAIN PYTHON TYPES, NOT ListValue
        logger.info("Setting user parameters (plain Python types)...")
        configs.seeds = seeds          # plain list
        configs.use_msa = use_msa      # plain bool
        configs.model.N_cycle = n_cycle
        configs.sample_diffusion.N_step = n_step
        configs.sample_diffusion.N_sample = n_sample
        apply_optional_request_configs(configs, request_params)
        
        logger.info(f"Config seeds type: {type(configs.seeds)}")
        logger.info(f"Config use_msa type: {type(configs.use_msa)}")
        
        # Download cache if needed (should be cached from init_handler)
        if not cache_downloaded:
            logger.info("Cache not downloaded yet, downloading now...")
            download_infercence_cache(configs)
        else:
            logger.info("Cache already downloaded during init")
        
        # ============================================================
        # RUN INFERENCE
        # ============================================================
        log_separator("INITIALIZING INFERENCE RUNNER")
        
        logger.info("Creating InferenceRunner...")
        runner = InferenceRunner(configs)
        logger.info("InferenceRunner created successfully!")
        
        logger.info("Starting infer_predict()...")
        infer_predict(runner, configs)
        logger.info("infer_predict() completed!")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
        
        # ============================================================
        # COLLECT RESULTS
        # ============================================================
        log_separator("COLLECTING RESULTS")
        
        output_files = collect_output_files(output_dir)
        
        if not output_files:
            logger.warning("No output files found after inference")
            return {"error": "Inference completed but no output files were generated"}
        
        log_separator(f"JOB {job_id} COMPLETED SUCCESSFULLY")
        logger.info(f"Returning {len(output_files)} files")
        
        return model_output_to_response(output_files)
    
    except Exception as e:
        logger.error(f"Error processing job: {e}")
        logger.error(traceback.format_exc())
        
        # Clear CUDA cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    
    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temp directory: {e}")


if __name__ == "__main__":
    init_handler()
    runpod.serverless.start({"handler": handler})
