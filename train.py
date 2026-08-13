import os

# Must be set BEFORE importing torch
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)

import gc
import logging
import shutil
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
import random

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("./dataset")
OUTPUT_DIR = Path("./outputs")
WEIGHTS_DIR = Path("./weights")

IMAGE_SIZE = 224

# Keep this True for maximum accuracy.
# Change to False only if you have a very large dataset.
USE_WIDER_BACKBONE = True

NUM_WORKERS = 4


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("DynamicPatchCore")


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================
# GPU INFORMATION
# ============================================================

def get_gpu_info():

    if not torch.cuda.is_available():
        return {
            "gpu": False,
            "name": "CPU",
            "total_gb": 0,
            "free_gb": 0,
        }

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    return {
        "gpu": True,
        "name": torch.cuda.get_device_name(0),
        "total_gb": total_bytes / (1024 ** 3),
        "free_gb": free_bytes / (1024 ** 3),
    }


# ============================================================
# DYNAMIC PATCHCORE CONFIGURATION
# ============================================================

def choose_configuration():

    # Force CPU mode by ignoring GPU checks
    info = {
        "gpu": False,
        "name": "CPU (Forced)",
        "total_gb": 32,
        "free_gb": 32,
    }

    if not info["gpu"]:

        logger.warning("Forcing CPU mode as requested.")

        return {
            "accelerator": "cpu",
            "batch_size": 4,  
            "eval_batch_size": 4,
            "backbone": "wide_resnet50_2", # Maximum accuracy backbone
            "layers": ["layer2", "layer3"],
            "coreset": 0.05, # Reduced from 0.10 to save RAM and time during k-center-greedy
            "image_size": 224, # Reduced from 256 to reduce feature map size
        }

    vram = info["total_gb"]

    logger.info(f"GPU: {info['name']}")
    logger.info(f"Total VRAM: {vram:.2f} GB")
    logger.info(f"Currently free VRAM: {info['free_gb']:.2f} GB")

    # --------------------------------------------------------
    # LARGE GPU
    # --------------------------------------------------------

    if vram >= 20:

        config = {
            "accelerator": "gpu",
            "batch_size": 8,
            "eval_batch_size": 8,
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset": 0.15,
            "image_size": 288,
        }

    # --------------------------------------------------------
    # 12-20 GB GPU
    # --------------------------------------------------------

    elif vram >= 12:

        config = {
            "accelerator": "gpu",
            "batch_size": 4,
            "eval_batch_size": 4,
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset": 0.15,
            "image_size": 288,
        }

    # --------------------------------------------------------
    # 8-12 GB GPU
    # --------------------------------------------------------

    elif vram >= 8:

        config = {
            "accelerator": "gpu",
            "batch_size": 2,
            "eval_batch_size": 2,
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "coreset": 0.15,
            "image_size": 256,
        }

    # --------------------------------------------------------
    # 6-8 GB GPU
    # --------------------------------------------------------

    elif vram >= 6:

        config = {
            "accelerator": "gpu",
            "batch_size": 1,
            "eval_batch_size": 1,
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "coreset": 0.10,
            "image_size": 256,
        }

    # --------------------------------------------------------
    # VERY SMALL GPU
    # --------------------------------------------------------

    else:

        config = {
            "accelerator": "gpu",
            "batch_size": 1,
            "eval_batch_size": 1,
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "coreset": 0.05,
            "image_size": 256,
        }

    return config


# ============================================================
# DATASET CHECK & SAMPLING
# ============================================================

def check_dataset():

    train_good = DATASET_ROOT / "train" / "good"
    sampled_dir = DATASET_ROOT / "train" / "good_sampled"

    logger.info("Checking dataset...")

    if not train_good.exists():
        raise FileNotFoundError(f"Missing dataset directory:\n{train_good}")

    all_images = [
        p for p in train_good.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]

    logger.info(f"Original dataset: {len(all_images)} images")

    if len(all_images) == 0:
        raise RuntimeError(f"No images found in {train_good}")

    # --------------------------------------------------------
    # MATHEMATICALLY SCALED FOR MAXIMUM STABILITY & ACCURACY:
    # Using 250 images with wide_resnet50_2 for maximum accuracy without crashing RAM.
    # --------------------------------------------------------
    max_images = 250
    
    sampled_dir.mkdir(parents=True, exist_ok=True)
    existing_sampled = list(sampled_dir.glob("*"))
    
    if len(existing_sampled) < min(max_images, len(all_images)):
        logger.info(f"Subsampling exactly {max_images} images for a 1-Day training schedule...")
        
        for f in existing_sampled:
            f.unlink()
            
        random.seed(42)
        sampled = random.sample(all_images, min(max_images, len(all_images)))
        
        for i, img in enumerate(sampled):
            shutil.copy2(img, sampled_dir / f"sampled_{i}{img.suffix}")
            
    logger.info(f"Successfully loaded {len(list(sampled_dir.glob('*')))} representative images for training.")


# ============================================================
# CREATE DATAMODULE
# ============================================================

def create_datamodule(config):

    logger.info("Creating datamodule...")

    datamodule = Folder(
        name="custom_dataset",
        root=DATASET_ROOT,

        # Normal training images ONLY (Using exactly 3000 subset)
        normal_dir="train/good_sampled",

        # Test images
        normal_test_dir="test/good",
        abnormal_dir="test/defect",

        train_batch_size=config["batch_size"],
        eval_batch_size=config["eval_batch_size"],

        num_workers=NUM_WORKERS,

        augmentations=v2.Compose([
            v2.Resize((config["image_size"], config["image_size"])),
            v2.ColorJitter(brightness=0.1, contrast=0.1)
        ]),
    )

    datamodule.setup()

    return datamodule


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(config):

    logger.info(
        f"Backbone: {config['backbone']}"
    )

    logger.info(
        f"Layers: {config['layers']}"
    )

    logger.info(
        f"Coreset ratio: {config['coreset']}"
    )

    model = Patchcore(
        backbone=config["backbone"],
        layers=config["layers"],
        pre_trained=True,
        coreset_sampling_ratio=config["coreset"],
        num_neighbors=9,
    )

    return model


# ============================================================
# CREATE ENGINE
# ============================================================

def create_engine(config):

    return Engine(
        default_root_dir=str(OUTPUT_DIR),

        accelerator=config["accelerator"],

        devices=1,

        # PatchCore only needs one pass.
        max_epochs=1,

        # Don't waste memory on sanity validation.
        num_sanity_val_steps=0,

        # Don't retain unnecessary computation graphs.
        inference_mode=True,
    )


# ============================================================
# TRAIN WITH AUTOMATIC OOM RECOVERY
# ============================================================

def train_dynamic(config):

    batch_sizes = []

    original_batch = config["batch_size"]

    # Automatically try smaller batches.
    current = original_batch

    while current >= 1:

        if current not in batch_sizes:
            batch_sizes.append(current)

        if current == 1:
            break

        current //= 2

    last_error = None

    for batch_size in batch_sizes:

        logger.info("=" * 60)
        logger.info(
            f"Trying batch size: {batch_size}"
        )
        logger.info("=" * 60)

        cleanup_memory()

        config["batch_size"] = batch_size
        config["eval_batch_size"] = min(
            batch_size,
            4
        )

        try:

            datamodule = create_datamodule(config)

            model = create_model(config)

            engine = create_engine(config)

            logger.info("Starting PatchCore training...")

            engine.fit(
                model=model,
                datamodule=datamodule,
            )

            logger.info("Training completed successfully.")

            logger.info("Starting evaluation...")

            engine.test(
                model=model,
                datamodule=datamodule,
            )

            logger.info("Evaluation completed.")

            return engine, model

        except torch.cuda.OutOfMemoryError as error:

            last_error = error

            logger.error(
                f"CUDA OOM at batch size {batch_size}"
            )

            cleanup_memory()

            if batch_size == 1:
                logger.error(
                    "Even batch size 1 caused CUDA OOM."
                )
                break

            continue

        except RuntimeError as error:

            message = str(error).lower()

            if (
                "out of memory" in message
                or "cuda" in message and "memory" in message
            ):

                last_error = error

                logger.error(
                    f"GPU memory error at batch size {batch_size}"
                )

                cleanup_memory()

                if batch_size == 1:
                    break

                continue

            raise

    # --------------------------------------------------------
    # GPU FAILED COMPLETELY -> CPU FALLBACK
    # --------------------------------------------------------

    logger.warning("=" * 60)
    logger.warning(
        "GPU configuration could not fit into memory."
    )
    logger.warning(
        "Falling back to CPU."
    )
    logger.warning("=" * 60)

    cleanup_memory()

    config["accelerator"] = "cpu"
    config["batch_size"] = 1
    config["eval_batch_size"] = 1

    # CPU fallback maintains high accuracy backbone.
    config["backbone"] = "wide_resnet50_2"
    config["coreset"] = min(
        config["coreset"],
        0.05
    )

    datamodule = create_datamodule(config)

    model = create_model(config)

    engine = create_engine(config)

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    engine.test(
        model=model,
        datamodule=datamodule,
    )

    return engine, model


# ============================================================
# FIND CHECKPOINT
# ============================================================

def find_checkpoint():

    if not OUTPUT_DIR.exists():
        return None

    checkpoints = list(
        OUTPUT_DIR.rglob("*.ckpt")
    )

    if not checkpoints:
        return None

    # Prefer files containing "best".
    best = [
        p for p in checkpoints
        if "best" in p.name.lower()
    ]

    if best:

        return max(
            best,
            key=lambda p: p.stat().st_mtime
        )

    return max(
        checkpoints,
        key=lambda p: p.stat().st_mtime
    )


# ============================================================
# SAVE WEIGHTS
# ============================================================

def save_weights():

    logger.info("Searching for checkpoint...")

    checkpoint = find_checkpoint()

    if checkpoint is None:

        logger.warning(
            "No .ckpt file found."
        )

        return None

    WEIGHTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        WEIGHTS_DIR /
        "patchcore_v2.ckpt"
    )

    shutil.copy2(
        checkpoint,
        destination
    )

    logger.info(
        f"Model saved to: {destination}"
    )

    return destination


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info("DYNAMIC PATCHCORE TRAINING")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    check_dataset()

    # --------------------------------------------------------
    # Select hardware configuration
    # --------------------------------------------------------

    config = choose_configuration()

    logger.info("=" * 60)
    logger.info("SELECTED CONFIGURATION")
    logger.info("=" * 60)

    for key, value in config.items():

        logger.info(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Tensor Core optimization
    # --------------------------------------------------------
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.set_float32_matmul_precision(
            "high"
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    engine, model = train_dynamic(config)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    checkpoint = save_weights()

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("TRAINING FINISHED")
    logger.info("=" * 70)

    if checkpoint:

        logger.info(
            f"FINAL MODEL: {checkpoint}"
        )

    else:

        logger.warning(
            "Training finished, but checkpoint was not found."
        )

    cleanup_memory()


if __name__ == "__main__":
    main()
