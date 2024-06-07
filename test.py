import cv2
import socket
import numpy as np
import struct
import time  # Import time for recording start time
from feature import calculate_compression_profile
import sys
import streamers.mjpeg as mjpeg

def cap_compression_profile(matrix):
    transformed_matrix = matrix * 100 * 15
    transformed_matrix = np.clip(transformed_matrix, 1, 50)
    transformed_matrix = transformed_matrix.astype(int)
    return transformed_matrix

def send_tile(client_socket, tile, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, tile_encoded = cv2.imencode('.jpg', tile, encode_param)
    tile_data = tile_encoded.tobytes()
    header = struct.pack('!I', len(tile_data))
    client_socket.sendall(header)
    client_socket.sendall(tile_data)

def send_image(client_socket, image, qualities):
    rows = len(qualities)
    cols = len(qualities[0])
    h, w, _ = image.shape
    tile_height, tile_width = h // rows, w // cols
    for i in range(rows):
        for j in range(cols):
            tile = image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]
            send_tile(client_socket, tile, qualities[i][j])

def test_read_frame():
    target_fps = 8.35

    cap = cv2.VideoCapture('videos/climbing.mp4')

    # Calculate the time to wait between frames
    frame_time = 1.0 / target_fps
    frames_read = 0
    test_start_time = time.time()

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        frames_read += 1

        if not ret:
            print("Failed to capture frame")
            print(f'Actual frame rate: {frames_read / (time.time() - test_start_time)}')
            break

        # Calculate elapsed time and sleep if necessary
        elapsed_time = time.time() - start_time
        time_to_wait = frame_time - elapsed_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)

def test_read_frame_and_compression():
    target_fps = 8.35

    cap = cv2.VideoCapture('videos/climbing.mp4')

    # Calculate the time to wait between frames
    frame_time = 1.0 / target_fps
    frames_read = 0
    test_start_time = time.time()

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        frames_read += 1

        if not ret:
            print("Failed to capture frame")
            print(f'Actual frame rate: {frames_read / (time.time() - test_start_time)}')
            break

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, frame_encoded = cv2.imencode('.jpg', frame, encode_param)

        # Calculate elapsed time and sleep if necessary
        elapsed_time = time.time() - start_time
        time_to_wait = frame_time - elapsed_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)

def test_read_frame_and_tile_compression():
    target_fps = 8.35

    cap = cv2.VideoCapture('videos/climbing.mp4')

    # Calculate the time to wait between frames
    frame_time = 1.0 / target_fps
    frames_read = 0
    test_start_time = time.time()

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        frames_read += 1

        if not ret:
            print("Failed to capture frame")
            print(f'Actual frame rate: {frames_read / (time.time() - test_start_time)}')
            break
        
        qualities = cap_compression_profile(calculate_compression_profile(frame, 2, 4))
            
        print("Compression Profile:\n", qualities)

        rows = len(qualities)
        cols = len(qualities[0])
        h, w, _ = frame.shape
        tile_height, tile_width = h // rows, w // cols
        for i in range(rows):
            for j in range(cols):
                tile = frame[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(qualities[i][j])]
                _, tile_encoded = cv2.imencode('.jpg', tile, encode_param)

        # Calculate elapsed time and sleep if necessary
        elapsed_time = time.time() - start_time
        time_to_wait = frame_time - elapsed_time
        if time_to_wait > 0:
            time.sleep(time_to_wait)

if __name__ == '__main__':
    # main()
    # test_read_frame()
    # test_read_frame_and_tile_compression()
    # test_read_frame_and_compression()