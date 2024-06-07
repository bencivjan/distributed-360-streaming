from flask import Flask, Response, url_for
import cv2
import numpy as np
import socket
import struct
import threading
import time  # Import time for recording frame times
from ultralytics import YOLO
from streamers import mjpeg, basic
from logger import Logger

app = Flask(__name__)

# Global variable to hold the latest image and frame times
video_captures = {}
frame_delays = {}  # Store frame delays
frame_times = {}  # Store frame times
frame_size = {} # Store size of frame after compression
# model_compute_times = {} # Store model compute times
# client_delays = {} # Store frame rendering times
# bandwidth = {}

def receive_tile(client_socket):
    header = client_socket.recv(4)
    tile_data_length = struct.unpack('!I', header)[0]

    tile_data = b''
    while len(tile_data) < tile_data_length:
        chunk = client_socket.recv(min(tile_data_length - len(tile_data), 4096))
        tile_data += chunk
    
    tile = cv2.imdecode(np.frombuffer(tile_data, np.uint8), cv2.IMREAD_COLOR)
    return tile, tile_data_length

def handle_client(client_socket, addr):
    global video_captures, frame_delays, frame_times, frame_size
    client_ip = addr[0]
    frame_delays[client_ip] = []
    frame_times[client_ip] = []
    frame_size[client_ip] = []

    while True:
        try:
            num_rows = struct.unpack('B', client_socket.recv(1))[0]
            num_cols = struct.unpack('B', client_socket.recv(1))[0]

            # Start time (first tile sent from client)
            frame_start_time = struct.unpack('!d', client_socket.recv(8))[0]

            # tiles = [receive_tile(client_socket) for _ in range(num_rows * num_cols)]
            compressed_frame_size = 0
            tiles = [None] * (num_rows * num_cols)
            for i in range(num_rows * num_cols):
                tile, compressed_tile_size = receive_tile(client_socket)
                tiles[i] = tile
                compressed_frame_size += compressed_tile_size
            frame_size[client_ip].append(compressed_frame_size)

            combined_rows = []
            index = 0
            for i in range(num_rows):
                row_tiles = [tiles[index + j] for j in range(num_cols)]
                combined_rows.append(np.hstack(row_tiles))
                index += num_cols
            video_captures[client_ip] = np.vstack(combined_rows)

            # End time (last tile received)
            frame_end_time = time.time()

            frame_delay = frame_end_time - frame_start_time
            frame_delays[client_ip].append(frame_delay)
            frame_times[client_ip].append(frame_end_time)

            # Print frame delay and FPS for each frame
            print(f"Frame Delay: {frame_delay:.6f} seconds")
            print(f"Compressed Frame Size: {frame_size[client_ip][-1] / 1000:2f} KB")
            if len(frame_times[client_ip]) > 1:
                fps = 1 / (frame_times[client_ip][-1] - frame_times[client_ip][-2])
                print(f"FPS: {fps:.2f}")

        except (ConnectionResetError, BrokenPipeError, struct.error):
            del video_captures[client_ip]
            print("Client disconnected or error occurred")
            break

    # Calculate and print the video processing statistics
    if client_ip in frame_delays:
        average_delay = sum(frame_delays[client_ip]) / len(frame_delays[client_ip])
        print(f"Average End-to-End Frame Delay: {average_delay:.6f} seconds")
    if client_ip in frame_times and len(frame_times[client_ip]) > 1:
        average_fps = len(frame_times[client_ip]) / (frame_times[client_ip][-1] - frame_times[client_ip][0])
        print(f"Average FPS: {average_fps:.2f}")
    if client_ip in frame_size and len(frame_size[client_ip]) > 1:
        average_frame_size = sum(frame_size[client_ip]) / len(frame_size[client_ip])
        print(f"Average Compressed Frame Size: {average_frame_size / 1000:2f} KB")

    # if client_ip in client_delays and len(client_delays[client_ip]) > 1:
    #     average_latency = sum(client_delays[client_ip]) / len(client_delays[client_ip])
    #     print(f"Average Render Latency: {average_latency:.6f} seconds")
    # if client_ip in bandwidth and len(bandwidth[client_ip]) > 1:
    #     average_bandwidth = sum(bandwidth[client_ip]) / len(bandwidth[client_ip])
    #     print(f"Average Bandwidth: {average_bandwidth:.6f} Mbps")
    # if client_ip in model_compute_times and len(model_compute_times[client_ip]) > 1:
    #     average_model_compute_time = sum(model_compute_times[client_ip]) / len(model_compute_times[client_ip])
    #     print(f"Average Bandwidth: {average_model_compute_time:.6f} seconds")

def video_feed(camera_id):
    global video_captures, client_delays, model_compute_times
    client_delays[camera_id] = []
    model_compute_times[camera_id] = []
    yolov8n_model = YOLO('yolov8l.pt')  # pretrained YOLOv8n model
    while True:
        start_time = time.time()
        if camera_id in video_captures:
            orig_img = video_captures[camera_id]
            start = time.time()
            results = yolov8n_model.predict(orig_img, verbose=False)
            print(f"Pred time: {time.time() - start}")
            annotated_frame = results[0].plot() # Visualize the results on the frame
            compute_time = results[0].speed['preprocess'] + results[0].speed['inference'] + results[0].speed['postprocess']
            model_compute_times[camera_id].append(compute_time)
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            # end_time = time.time()
            # latency = end_time - start_time
            # client_delays[camera_id].append(latency)
            # print(f"Frame render latency for camera {camera_id}: {latency:.6f}")

@app.route(f'/video_feed/<string:camera_id>')
def video_feed_route(camera_id):
    return Response(video_feed(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    global video_captures
    links = ''
    for camera_id in video_captures.keys():
        links += f'<p><a href="{url_for("video_feed_route", camera_id=camera_id)}">{camera_id}</a></p>'
    return links

def main():
    HOST_PUBLIC = '0.0.0.0'
    HOST_LOCAL = 'localhost'
    SOCKET_PORT = 8010
    WEB_PORT = 8080
    server_socket = socket.socket()
    server_socket.bind((HOST_PUBLIC, SOCKET_PORT))
    server_socket.listen(1)
    # So we don't have to wait when restarting the server
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    threading.Thread(target=app.run, kwargs={'host':HOST_PUBLIC, 'port':WEB_PORT}).start()

    while True:
        client_socket, addr = server_socket.accept()
        threading.Thread(target=handle_client, kwargs={'client_socket':client_socket, 'addr': addr}).start()

def test():
    HOST_PUBLIC = '0.0.0.0'
    HOST_LOCAL = 'localhost'
    SOCKET_PORT = 8010
    WEB_PORT = 8080
    server_socket = socket.socket()
    server_socket.bind((HOST_PUBLIC, SOCKET_PORT))
    server_socket.listen(1)
    # So we don't have to wait when restarting the server
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    print(f'==TEST==\nListening on port {SOCKET_PORT}')
    while True:
        client_socket, addr = server_socket.accept()
        print(f'==NEW CONNECTION==\n{addr}')
        threading.Thread(target=handle_client, kwargs={'client_socket':client_socket, 'addr': addr}).start()

def test_stream_frame():
    HOST_PUBLIC = '0.0.0.0'
    HOST_LOCAL = 'localhost'
    SOCKET_PORT = 8010
    WEB_PORT = 8080
    server_socket = socket.socket()
    server_socket.bind((HOST_PUBLIC, SOCKET_PORT))
    server_socket.listen(1)
    # So we don't have to wait when restarting the server
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def handle_client_frame(client_socket, addr):
        client_ip = addr[0]
        frame_delays[client_ip] = []
        frame_times[client_ip] = []
        frame_size[client_ip] = []

        logger = Logger('./mjpeg-logs.txt')
        streamer = mjpeg.Mjpeg(client_socket, logger=logger)

        while True:
            try:
                frame = streamer.get_frame()

            except (ConnectionResetError, BrokenPipeError, struct.error):
                print("Client disconnected or error occurred")
                break
        
        logger.flush()
        
    print(f'==TEST WITH NO COMPRESSION==\nListening on port {SOCKET_PORT}')
    while True:
        client_socket, addr = server_socket.accept()
        print(f'==NEW CONNECTION==\n{addr}')
        threading.Thread(target=handle_client_frame, kwargs={'client_socket':client_socket, 'addr': addr}).start()

if __name__ == '__main__':
    #main()
    # test()
    test_stream_frame()