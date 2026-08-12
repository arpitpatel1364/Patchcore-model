import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from pathlib import Path
import torchvision.transforms.v2 as v2
import shutil
import logging
import torch

def main():
    # Set matrix multiplication precision for RTX Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # Setup logging to show proper logs in terminal
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define dataset paths
    dataset_root = Path("./dataset")
    
    # 1. Setup Dataset (High Accuracy settings)
    datamodule = Folder(
        name="custom_dataset",
        root=dataset_root,
        normal_dir="train/good",
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        train_batch_size=8,  # Increased back for CPU
        eval_batch_size=8,
        num_workers=4,
        augmentations=v2.Resize((256, 256))  # Restored high resolution
    )
    datamodule.setup()

    # 2. Setup Model (PatchCore with ResNet18)
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1  # Restored high retention for best accuracy
    )

    # 3. Setup Engine (Using CPU to bypass GPU memory limits entirely)
    engine = Engine(
        default_root_dir="./outputs",
        accelerator="cpu",  # <--- Changed to CPU
        devices=1
    )

    # 4. Train the model
    print("Starting training...")
    engine.fit(datamodule=datamodule, model=model)
    print("Training completed.")
    
    # 5. Evaluate the model
    print("Starting evaluation...")
    engine.test(datamodule=datamodule, model=model)
    
    # 6. Copy the model to weights directory
    print("Locating the final trained weights...")
    ckpt_files = list(Path("./outputs").rglob("*.ckpt"))
    if ckpt_files:
        # Get the most recently created checkpoint
        latest_ckpt = max(ckpt_files, key=os.path.getctime)
        weights_dir = Path("./weights")
        weights_dir.mkdir(exist_ok=True)
        dest_path = weights_dir / "patchcore_best.ckpt"
        
        shutil.copy2(latest_ckpt, dest_path)
        print(f"✅ Success! Model weights copied to: {dest_path}")
    else:
        print("⚠️ Warning: No checkpoint files found to copy.")
    
if __name__ == "__main__":
    main()
