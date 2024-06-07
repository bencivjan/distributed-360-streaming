import cv2
import socket
import numpy as np
import struct
import time  # Import time for recording start time
from feature import calculate_compression_profile
import sys
from streamers import mjpeg, basic

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

    cap = cv2.VideoCapture('videos/climbing.mp4')
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

def test_stream_frame(compression='none'):
    print(f'STARTING {compression.upper()} COMPRESSION TEST')
    if len(sys.argv) < 2:
        # TCP_IP = '130.126.136.178' # default server address
        # TCP_IP = '100.72.81.13'
        print('Input server IP address as first argument')
        return
    else:
        TCP_IP = sys.argv[1]
    TCP_PORT = 8010

    cap = cv2.VideoCapture('videos/climbing.mp4')
    client_socket = socket.socket()
    client_socket.connect((TCP_IP, TCP_PORT))

    try:
        # target_fps = 5 # Thottle to set fps for energy consistency
        # Calculate the time to wait between frames
        # frame_time = 1.0 / target_fps
        frames_read = 0
        test_start_time = time.time()

        print(f'Streaming with {compression} compression')
        if compression == 'none':
            client_socket.sendall(struct.pack('B', 0x0))
            streamer = basic.Basic(client_socket)
        elif compression == 'jpeg-30':
            client_socket.sendall(struct.pack('B', 0x1))
            streamer = mjpeg.Mjpeg(client_socket, qf=30)
        elif compression == 'jpeg-50':
            client_socket.sendall(struct.pack('B', 0x2))
            streamer = mjpeg.Mjpeg(client_socket, qf=50)
        elif compression == 'jpeg-90':
            client_socket.sendall(struct.pack('B', 0x3))
            streamer = mjpeg.Mjpeg(client_socket, qf=90)
        else:
            print('Unsupported compression algorithm!')
            return

        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            frames_read += 1
            if not ret:
                print("Failed to capture frame")
                print(f'Actual frame rate: {frames_read / (time.time() - test_start_time)}')
                break

            streamer.send_frame(frame)
            print(frame.nbytes)

            # Calculate elapsed time and sleep if necessary
            # elapsed_time = time.time() - start_time
            # time_to_wait = frame_time - elapsed_time
            # if time_to_wait > 0:
            #     time.sleep(time_to_wait)
    finally:
        cap.release()
        client_socket.close()

if __name__ == '__main__':
    test_stream_frame(compression='jpeg-50')