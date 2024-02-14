import cv2
import socket
import numpy as np
import struct
import time  # Import time for recording start time
from feature import calculate_edge_density, calculate_corner_density, calculate_contour_density

def calculate_compression_profile(frame, num_rows, num_cols):
    edge_density = calculate_edge_density(frame, num_rows, num_cols)
    corner_density = calculate_corner_density(frame, num_rows, num_cols)
    contour_density = calculate_contour_density(frame, num_rows, num_cols)

    combined_density = np.array([edge_density, corner_density, contour_density])
    average_density = np.mean(combined_density, axis=0)

    return average_density

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
    TCP_IP = '10.251.13.228'# '100.72.81.20' # 130.126.136.178'
    TCP_PORT = 8010

    cap = cv2.VideoCapture('/home/bencivjan/Desktop/climbing.mp4')
    client_socket = socket.socket()
    client_socket.connect((TCP_IP, TCP_PORT))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

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
            
            send_image(client_socket, frame, qualities)
    finally:
        cap.release()
        client_socket.close()

if __name__ == '__main__':
    main()
