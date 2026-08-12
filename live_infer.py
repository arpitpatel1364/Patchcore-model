import cv2
import sys
import os
import numpy as np
import torch
from anomalib.deploy import TorchInferencer
import tkinter as tk
from tkinter import filedialog

os.environ["TRUST_REMOTE_CODE"] = "1"

def main():
    if len(sys.argv) < 2:
        # Hide the main tkinter window
        root = tk.Tk()
        root.withdraw()
        print("Please select a video file from the dialog...")
        video_source = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
        )
        if not video_source:
            print("No file selected. Exiting.")
            return
    else:
        video_source = sys.argv[1]
    
    # The exported PyTorch model from anomalib export
    checkpoint_path = "results/weights/torch/model.pt"
    
    print(f"Loading AI Model from {checkpoint_path}...")
    try:
        inferencer = TorchInferencer(
            path=checkpoint_path,
            device="cpu"  # Forcefully run on CPU to avoid VRAM crashes
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    print("Model loaded successfully!")
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
        
    # Setup Video Writer to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_filename = f"processed_{video_source}"
    out_video = cv2.VideoWriter(out_filename, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    print(f"Processing video '{video_source}'...")
    print("Please wait. This might take a while because the video is over 1GB.")
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a resizable window that fits the screen
    cv2.namedWindow("Live Detections", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Detections", 960, 540)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\nVideo processing complete! Saved as '{out_filename}'")
            break
            
        frame_count += 1
        if frame_count % 15 == 0:
            print(f"Processed {frame_count} / {total_frames} frames...", end='\r')
            
        # Convert BGR to RGB for the AI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = inferencer.predict(image=rgb_frame)
        
        score = predictions.pred_score
        
        # Ensure it's a numeric score if it's a tensor
        if isinstance(score, torch.Tensor):
            score = score.item()
            
        anomaly_map = predictions.anomaly_map
        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.detach().cpu().numpy()
            
        anomaly_map = np.squeeze(anomaly_map)
        
        # Try to use the model's built-in prediction mask if available
        if hasattr(predictions, 'pred_mask') and predictions.pred_mask is not None:
            pred_mask = predictions.pred_mask
            if isinstance(pred_mask, torch.Tensor):
                pred_mask = pred_mask.detach().cpu().numpy()
            pred_mask = np.squeeze(pred_mask)
            # Ensure it is an 8-bit image for OpenCV
            if pred_mask.max() <= 1:
                thresh = (pred_mask * 255).astype(np.uint8)
            else:
                thresh = pred_mask.astype(np.uint8)
            # Resize mask to match frame
            thresh = cv2.resize(thresh, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            # Fallback if pred_mask isn't available: use a stricter threshold on the anomaly map
            heatmap = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            # Use a much stricter threshold (e.g. 200 instead of 128) to avoid background noise
            _, thresh = cv2.threshold(heatmap, 200, 255, cv2.THRESH_BINARY)
            
        status = "DEFECT DETECTED" if score > 0.5 else "GOOD"
        color = (0, 0, 255) if status == "DEFECT DETECTED" else (0, 255, 0)
        
        annotated_frame = frame.copy()
        
        if status == "DEFECT DETECTED":
            # Find exact contours of the defects
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw the exact contours instead of squares!
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:  # Ignore tiny specks
                    # Draw smooth contours in red
                    cv2.drawContours(annotated_frame, [cnt], -1, (0, 0, 255), 3)
        
        cv2.putText(annotated_frame, f"{status} (Score: {score:.2f})", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    
        out_video.write(annotated_frame)
        
        # Show live video window
        cv2.imshow("Live Detections", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nLive detection stopped by user.")
            break
            
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
