from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder
import torchvision.transforms.v2 as v2
from pathlib import Path

def main():
    # 1. Load Model Setup
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.05
    )
    
    engine = Engine(accelerator="auto", devices=1)
    
    checkpoint_path = "./weights/patchcore_best.ckpt" 
    
    if not Path(checkpoint_path).exists():
        print(f"Cannot find model file at: {checkpoint_path}")
        return

    # 2. Setup the Exact Same Folder Datamodule used in train.py
    # This completely bypasses PyTorch collation errors because it uses Anomalib's native pipeline!
    datamodule = Folder(
        name="test_dataset",
        root="./dataset",
        normal_dir="train/good",
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        eval_batch_size=1,
        augmentations=v2.Resize((224, 224))
    )
    datamodule.setup()

    # 3. Run Inference using the tested pipeline
    print("Running evaluation on the test images...")
    try:
        engine.test(
            model=model,
            ckpt_path=checkpoint_path,
            datamodule=datamodule
        )
        print("Inference completed successfully.")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
