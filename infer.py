from anomalib.engine import Engine
from anomalib.models import Patchcore
from pathlib import Path

def main():
    # 1. Initialize model with the same backbone used during training
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1
    )
    
    # 2. Setup Engine
    engine = Engine(
        accelerator="auto",
        devices=1
    )
    
    # Update this path with the actual checkpoint generated in outputs/ during training
    # Anomalib automatically creates timestamped or versioned folders inside outputs/
    checkpoint_path = "./outputs/Patchcore/custom_dataset/latest/weights/lightning_logs/version_0/checkpoints/last.ckpt" 
    
    # Path to test image (dynamically pick the first defect image)
    defect_dir = Path("./dataset/test/defect")
    image_list = list(defect_dir.glob("*.jpg"))
    
    if not defect_dir.exists() or not image_list:
        print(f"No images found in {defect_dir}. Please provide a valid image for inference.")
        return
        
    image_path = str(image_list[0])

    # 3. Run Inference
    print(f"Running inference on {image_path}...")
    try:
        predictions = engine.predict(
            model=model,
            ckpt_path=checkpoint_path,
            data_paths=[image_path]
        )
        print("Inference completed successfully.")
        print(predictions)
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Please ensure you have trained the model first and the checkpoint path is correct.")

if __name__ == "__main__":
    main()
