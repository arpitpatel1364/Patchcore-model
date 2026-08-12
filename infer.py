from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder
import torchvision.transforms.v2 as v2
from pathlib import Path

def main():
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.05
    )
    
    engine = Engine(accelerator="auto", devices=1)
    
    checkpoint_path = "./weights/patchcore_best.ckpt" 
    
    defect_dir = Path("./dataset/test/defect")
    if not defect_dir.exists():
        print(f"No images found in {defect_dir}.")
        return

    # Use Folder datamodule for automatic resizing and proper PyTorch collation
    datamodule = Folder(
        name="predict_dataset",
        root="./dataset",
        normal_dir="train/good",
        predict_dir="test/defect",
        predict_batch_size=1,
        augmentations=v2.Resize((224, 224)) # Critical: MUST match train.py!
    )
    datamodule.setup()

    print(f"Running inference on images in {defect_dir}...")
    try:
        predictions = engine.predict(
            model=model,
            ckpt_path=checkpoint_path,
            datamodule=datamodule
        )
        print("Inference completed successfully.")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
