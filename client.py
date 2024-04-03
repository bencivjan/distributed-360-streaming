import cv2
import socket
import numpy as np
import struct
import time  # Import time for recording start time
from feature import calculate_compression_profile
import sys
from ultralytics import YOLO
import torch

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

def main():
    if len(sys.argv) < 2:
        TCP_IP = '130.126.136.178' # default server address
    else:
        TCP_IP = sys.argv[1]
    TCP_PORT = 8010

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('../climbing.mp4')
    client_socket = socket.socket()
    client_socket.connect((TCP_IP, TCP_PORT))
    predict_latency = [] # Store model prediction latency
    yolov8n_model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            start_time = time.time()
            results = yolov8n_model.predict(frame, verbose=False)[0]
            end_time = time.time()
            print(f"Time to analyze frame: {(end_time - start_time):.6f}s")
            predict_latency.append(end_time - start_time)
            blurred_frame = cv2.GaussianBlur(frame, (0, 0), 15)
            print(f"results.boxes {results.boxes}")
            mask = np.zeros_like(frame)
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = box.to(torch.int)
                mask[y1:y2, x1:x2] = [255, 255, 255]

            # annotated_frame = results.plot() # Visualize the results on the frame
            annotated_frame = np.where(mask==[255, 255, 255], frame, blurred_frame)
            print(results)

            # qualities = [[100, 100, 100, 100], [100, 100, 100, 100]]
            qualities = cap_compression_profile(calculate_compression_profile(frame, 2, 4))
            
            print("Compression Profile:\n", qualities)
            # Send the number of rows and columns
            num_rows, num_cols = len(qualities), len(qualities[0])
            client_socket.sendall(struct.pack('B', num_rows))
            client_socket.sendall(struct.pack('B', num_cols))
            
            # Send the timestamp
            timestamp = time.time()
            client_socket.sendall(struct.pack('!d', timestamp))
            
            send_image(client_socket, annotated_frame, qualities)
    finally:
        print(f"Average model inference time: {np.mean(predict_latency):.6f}s")
        cap.release()
        client_socket.close()

if __name__ == '__main__':
    main()
