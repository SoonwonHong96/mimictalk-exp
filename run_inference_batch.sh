#!/bin/bash

# This script runs the mimictalk inference on all mp4 files in a directory.

VIDEO_DIR="data/raw/examples"
SCRIPT_PATH="inference/train_mimictalk_on_a_video.py"

# Check if the video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
  echo "Error: Video directory not found at $VIDEO_DIR"
  exit 1
fi

# Find all mp4 files in the directory
VIDEO_FILES=$(find "$VIDEO_DIR" -type f -name "*.mp4")

if [ -z "$VIDEO_FILES" ]; then
    echo "No .mp4 files found in $VIDEO_DIR"
    exit 0
fi

for video_file in $VIDEO_FILES
do
  echo "----------------------------------------------------"
  echo "Processing video: $video_file"
  echo "----------------------------------------------------"
  python "$SCRIPT_PATH" --lora_r 2 --video_id "$video_file"
  
  # Check the exit code of the python script
  if [ $? -ne 0 ]; then
    echo "Error processing $video_file. Exiting."
    exit 1
  fi
done

echo "----------------------------------------------------"
echo "All videos processed successfully."
echo "----------------------------------------------------" 