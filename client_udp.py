import cv2
import socket
import numpy as np
import struct
import time  # Import time for recording start time
from feature import calculate_compression_profile
import sys

def cap_compression_profile(matrix):
    transformed_matrix = matrix * 100 * 15
    # Clip at 40 to prevent UDP packet overflow, but may not work for all videos
    # Better solution is to hard clip in send_tile_udp function
    transformed_matrix = np.clip(transformed_matrix, 1, 40)
    transformed_matrix = transformed_matrix.astype(int)
    return transformed_matrix

def send_tile_udp(client_socket, UDP_ADDR, tile, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, tile_encoded = cv2.imencode('.jpg', tile, encode_param)
    tile_data = tile_encoded.tobytes()
    header = struct.pack('!I', len(tile_data))
    client_socket.sendto(header, UDP_ADDR)
    client_socket.sendto(tile_data, UDP_ADDR)

def send_image_udp(client_socket, UDP_ADDR, image, qualities):
    rows = len(qualities)
    cols = len(qualities[0])
    h, w, _ = image.shape
    tile_height, tile_width = h // rows, w // cols
    for i in range(rows):
        for j in range(cols):
            tile = image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]
            send_tile_udp(client_socket, UDP_ADDR, tile, qualities[i][j])

def main():
    if len(sys.argv) < 2:
        UDP_IP = '130.126.136.178' # default server address
    else:
        UDP_IP = sys.argv[1]
    UDP_PORT = 8010
    UDP_ADDR = (UDP_IP, UDP_PORT)

    cap = cv2.VideoCapture('../climbing.mp4')
    client_socket = socket.socket(socket.AF_INET, # Internet
                                  socket.SOCK_DGRAM) # UDP

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # If we reach the end of the video, reset to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("Finished video stream, restarting from beginning")
                continue

            # qualities = [[100, 100, 100, 100], [100, 100, 100, 100]]
            qualities = cap_compression_profile(calculate_compression_profile(frame, 2, 4))
            
            print("Compression Profile:\n", qualities)
            # Send the number of rows and columns
            num_rows, num_cols = len(qualities), len(qualities[0])
            client_socket.sendto(struct.pack('B', num_rows), UDP_ADDR)
            client_socket.sendto(struct.pack('B', num_cols), UDP_ADDR)
            
            # Send the timestamp
            timestamp = time.time()
            client_socket.sendto(struct.pack('!d', timestamp), UDP_ADDR)
            
            send_image_udp(client_socket, UDP_ADDR, frame, qualities)
    finally:
        cap.release()
        client_socket.close()

if __name__ == '__main__':
    main()
