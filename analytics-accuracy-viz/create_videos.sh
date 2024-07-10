#!/bin/bash

# Source video file from https://www.kaggle.com/datasets/smeschke/pedestrian-dataset/data
# input_video="videos/crosswalk.avi"
input_video="videos/ny_driving.mov"

# Array of bitrate and frame rate combinations to encode
# bitrates=("250K" "500k" "750K" "1m")
bitrates=("1m" "3m" "5m")
framerates=("10" "15" "30")

# Loop through each combination of bitrate and frame rate
for bitrate in "${bitrates[@]}"; do
    for framerate in "${framerates[@]}"; do
        # Output file name based on bitrate and framerate
        output_file="nydriving_${bitrate}_${framerate}.mp4"

        # FFmpeg command to transcode
        ffmpeg -i "$input_video" -s 1920x1080 -vf "fps=$framerate" -b:v "$bitrate" -c:v libx264 "$output_file"
    done
done
