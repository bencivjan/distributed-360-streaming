from flask import Flask, Response, url_for
import cv2
import numpy as np
import socket
import struct
import threading
import time  # Import time for recording frame times

app = Flask(__name__)

# Global variable to hold the latest image and frame times
video_captures = {}
frame_delays = {}  # Store frame delays
frame_times = {}  # Store frame times

def receive_tile(client_socket):
    header = client_socket.recv(4)
    tile_data_length = struct.unpack('!I', header)[0]

    tile_data = b''
    while len(tile_data) < tile_data_length:
        chunk = client_socket.recv(min(tile_data_length - len(tile_data), 4096))
        tile_data += chunk
    
    tile = cv2.imdecode(np.frombuffer(tile_data, np.uint8), cv2.IMREAD_COLOR)
    return tile

def handle_client(client_socket, addr):
    global video_captures, frame_delays, frame_times
    client_ip = addr[0]
    frame_delays[client_ip] = []
    frame_times[client_ip] = []

    while True:
        try:
            num_rows = struct.unpack('B', client_socket.recv(1))[0]
            num_cols = struct.unpack('B', client_socket.recv(1))[0]

            # Start time (first tile sent from client)
            frame_start_time = struct.unpack('!d', client_socket.recv(8))[0]

            tiles = [receive_tile(client_socket) for _ in range(num_rows * num_cols)]

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
            if len(frame_times[client_ip]) > 1:
                fps = 1 / (frame_times[client_ip][-1] - frame_times[client_ip][-2])
                print(f"FPS: {fps:.2f}")

        except (ConnectionResetError, BrokenPipeError, struct.error):
            print("Client disconnected or error occurred")
            break

    # Calculate and print the average end-to-end frame delay and FPS
    if client_ip in frame_delays:
        average_delay = sum(frame_delays[client_ip]) / len(frame_delays[client_ip])
        print(f"Average End-to-End Frame Delay: {average_delay:.6f} seconds")
    if client_ip in frame_times and len(frame_times[client_ip]) > 1:
        average_fps = len(frame_times[client_ip]) / (frame_times[client_ip][-1] - frame_times[client_ip][0])
        print(f"Average FPS: {average_fps:.2f}")

def video_feed(camera_id):
    global video_captures
    while True:
        if camera_id in video_captures:
            ret, jpeg = cv2.imencode('.jpg', video_captures[camera_id])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route(f'/video_feed/<string:camera_id>')
def video_feed_route(camera_id):
    return Response(video_feed(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
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

    threading.Thread(target=app.run, kwargs={'host':HOST_PUBLIC, 'port':WEB_PORT}).start()

    while True:
        client_socket, addr = server_socket.accept()
        threading.Thread(target=handle_client, kwargs={'client_socket':client_socket, 'addr': addr}).start()

if __name__ == '__main__':
    main()
