import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, frame_interval=1, max_frames=None, current_total=0, target_fps=None):
    """
    Extracts frames from a video and saves them to the output directory.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where frames will be saved.
        frame_interval (int): Save every Nth frame.
        max_frames (int): Maximum number of frames to save across all videos.
        current_total (int): Current count of frames saved before this video.
        target_fps (float): Extract frames at a specific rate (e.g., 2.0). Overrides frame_interval.
        
    Returns:
        int: The updated count of total frames saved.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return current_total

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing '{video_path.name}' | FPS: {fps:.2f} | Total Frames: {total_frames}")

    if target_fps is not None and target_fps > 0:
        frame_interval = max(1, int(fps / target_fps))
        print(f"Target FPS: {target_fps} -> Extracting every {frame_interval} frames.")

    frame_count = 0
    saved_count = 0
    
    video_prefix = video_path.stem

    while True:
        if max_frames is not None and current_total >= max_frames:
            print("Reached the maximum frame limit.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = output_dir / f"{video_prefix}_frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
            current_total += 1
            
        frame_count += 1

    cap.release()
    print(f"-> Saved {saved_count} frames from this video. Total saved so far: {current_total}\n")
    return current_total

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos up to a limit.")
    parser.add_argument("--input_dir", type=str, default="./dataset", help="Directory containing the video files.")
    parser.add_argument("--output_dir", type=str, default="./dataset/train/good", help="Directory to save the extracted frames.")
    parser.add_argument("--interval", type=int, default=1, help="Extract every Nth frame.")
    parser.add_argument("--target_fps", type=float, default=None, help="Extract frames at this specific rate (e.g. 2). Overrides --interval.")
    parser.add_argument("--ext", type=str, default="mp4", help="Video file extension to look for.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to extract overall (e.g., 5000).")
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return

    video_files = list(input_dir.glob(f"*.{args.ext}")) + list(input_dir.glob(f"*.{args.ext.upper()}"))
    
    if not video_files:
        print(f"No .{args.ext} video files found in {input_dir}.")
        return

    total_extracted = 0
    for video_file in video_files:
        if args.max_frames is not None and total_extracted >= args.max_frames:
            break
        total_extracted = extract_frames(
            video_file, 
            output_dir, 
            frame_interval=args.interval, 
            max_frames=args.max_frames,
            current_total=total_extracted,
            target_fps=args.target_fps
        )
        
    print(f"All done! Total images ready in {output_dir}: {total_extracted}")

if __name__ == "__main__":
    main()
