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
import os

app = Flask(__name__)

# Global variable to hold the latest image and frame times
video_captures = {}

def handle_client(client_socket, addr):
    global video_captures
    client_ip = addr[0]

    IMGS_PATH = f'./received_imgs_{client_ip}/'
    os.makedirs(IMGS_PATH, exist_ok=True)
    frame_idx = 0

    logger = Logger(f'./mjpeg_logs_{client_ip}.txt')

    compression_alg = struct.unpack('B', client_socket.recv(1))[0]
    print(f'compression_alg: {compression_alg}')
    if compression_alg == 0x0:
        streamer = basic.Basic(client_socket, logger=logger)
    elif compression_alg in [0x1, 0x2, 0x3]:
        streamer = mjpeg.Mjpeg(client_socket, logger=logger)
    else:
        print('Unsupported compression algorithm!')
        return

    while True:
        try:
            frame = streamer.get_frame()
            video_captures[client_ip] = frame
            img_name = IMGS_PATH + str(frame_idx) + '.jpg'
            ret = cv2.imwrite(img_name, frame)
            frame_idx += 1
            if ret == False:
                print(f'Failed to write image to {img_name}')
        except (ConnectionResetError, BrokenPipeError, struct.error):
            print("Client disconnected or error occurred")
            break
    logger.flush()

def video_feed(camera_id):
    global video_captures, client_delays, model_compute_times
    client_delays[camera_id] = []
    model_compute_times[camera_id] = []
    yolov8n_model = YOLO('yolov8l.pt')  # pretrained YOLOv8n model
    while True:
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

if __name__ == '__main__':
    main()