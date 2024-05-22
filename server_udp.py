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
client_latency = {} # Store frame rendering times
bandwidth = {}

def recv_bytes_udp(sock, buf_len, max_recv_bound=float('inf')):
    buf_bytes = b''
    addr = ''
    while len(buf_bytes) < buf_len:
        # try:
        chunk, addr = sock.recvfrom(min(buf_len - len(buf_bytes), max_recv_bound))
        if not chunk:
            # Handle the case where the connection is closed
            raise socket.error("Connection closed prematurely")
        buf_bytes += chunk
        # except (KeyboardInterrupt):
        #     print("Server terminated with KeyboardInterrupt")
        #     break
    return buf_bytes, addr

def receive_tile(server_socket):
    header, _ = recv_bytes_udp(server_socket, 4)
    tile_data_length = struct.unpack('!I', header)[0]
    tile_data, _ = recv_bytes_udp(server_socket, tile_data_length)
    tile = cv2.imdecode(np.frombuffer(tile_data, np.uint8), cv2.IMREAD_COLOR)
    return tile

def handle_client(sock):
    global video_captures, frame_delays, frame_times, client_latency, bandwidth

    while True:
        try:
            # Get client ip from first byte read, find a better way
            # to handle this for multiple clients
            num_rows_bytes, (client_ip, client_port) = recv_bytes_udp(sock, 1)
            num_rows = struct.unpack('B', num_rows_bytes)[0]
            num_cols = struct.unpack('B', recv_bytes_udp(sock, 1)[0])[0]

            if client_ip not in frame_delays:
                frame_delays[client_ip] = []
            if client_ip not in frame_times:
                frame_times[client_ip] = []
            if client_ip not in client_latency:
                client_latency[client_ip] = []
            if client_ip not in bandwidth:
                bandwidth[client_ip] = []

            # Start time (first tile sent from client)
            frame_start_time = struct.unpack('!d', recv_bytes_udp(sock, 8)[0])[0]

            tiles = [receive_tile(sock) for _ in range(num_rows * num_cols)]

            combined_rows = []
            index = 0
            try:
                for i in range(num_rows):
                    row_tiles = [tiles[index + j] for j in range(num_cols)]
                    combined_rows.append(np.hstack(row_tiles))
                    index += num_cols
                video_captures[client_ip] = np.vstack(combined_rows)
            except ValueError as v:
                print(f"Error stacking tiles: {v}")

            # End time (last tile received)
            frame_end_time = time.time()

            frame_delay = frame_end_time - frame_start_time
            frame_delays[client_ip].append(frame_delay)
            frame_times[client_ip].append(frame_end_time)
            bandwidth[client_ip].append((video_captures[client_ip].nbytes) / (frame_delay * 1024 * 1024))

            # Print frame delay and FPS for each frame
            print(f"Frame Delay: {frame_delay:.6f} seconds")
            print(f"Bandwidth: {bandwidth[client_ip][-1]:.2f} Mbps")
            if len(frame_times[client_ip]) > 1:
                fps = 1 / (frame_times[client_ip][-1] - frame_times[client_ip][-2])
                print(f"FPS: {fps:.2f}")

        except (KeyboardInterrupt, ConnectionResetError, BrokenPipeError, struct.error):
            del video_captures[client_ip]
            print("Client disconnected or error occurred")
            break

    # Calculate and print the average end-to-end frame delay and FPS
    if client_ip in frame_delays:
        average_delay = sum(frame_delays[client_ip]) / len(frame_delays[client_ip])
        print(f"Average End-to-End Frame Delay: {average_delay:.6f} seconds")
    if client_ip in frame_times and len(frame_times[client_ip]) > 1:
        average_fps = len(frame_times[client_ip]) / (frame_times[client_ip][-1] - frame_times[client_ip][0])
        print(f"Average FPS: {average_fps:.2f}")
    if client_ip in client_latency and len(client_latency[client_ip]) > 1:
        average_latency = sum(client_latency[client_ip]) / len(client_latency[client_ip])
        print(f"Average Render Latency: {average_latency:.6f} seconds")
    if client_ip in bandwidth and len(bandwidth[client_ip]) > 1:
        average_bandwidth = sum(bandwidth[client_ip]) / len(bandwidth[client_ip])
        print(f"Average Bandwidth: {average_bandwidth:.6f} seconds")

def video_feed(camera_id):
    global video_captures, client_latency
    client_latency[camera_id] = []
    while True:
        start_time = time.time()
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
    sock = socket.socket(socket.AF_INET, # Internet
                                  socket.SOCK_DGRAM) # UDP
    sock.bind((HOST_PUBLIC, SOCKET_PORT))

    threading.Thread(target=app.run, kwargs={'host':HOST_PUBLIC, 'port':WEB_PORT}).start()

    # Note: This can only properly handle one client currently
    handle_client(sock)

if __name__ == '__main__':
    main()
