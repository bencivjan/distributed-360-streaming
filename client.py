import cv2
import socket
import numpy as np
import struct
import time
import sys
import os
from streamers import mjpeg, basic, tile_spatial

mod_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'streamers', 'ffenc_uiuc'))
if mod_dir not in sys.path:
    sys.path.append(mod_dir)
from streamers.ffenc_uiuc import h264

def stream_video(compression='none'):
    print(f'STARTING {compression.upper()} COMPRESSION TEST')
    if len(sys.argv) < 2:
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
        elif compression == 'mjpeg-30':
            client_socket.sendall(struct.pack('B', 0x1))
            streamer = mjpeg.Mjpeg(client_socket, qf=30)
        elif compression == 'mjpeg-50':
            client_socket.sendall(struct.pack('B', 0x2))
            streamer = mjpeg.Mjpeg(client_socket, qf=50)
        elif compression == 'mjpeg-90':
            client_socket.sendall(struct.pack('B', 0x3))
            streamer = mjpeg.Mjpeg(client_socket, qf=90)
        elif compression == 'tiled-spatial':
            client_socket.sendall(struct.pack('B', 0x4))
            streamer = tile_spatial.TileSpatial(client_socket)
        elif compression == 'h264':
            client_socket.sendall(struct.pack('B', 0x5))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            streamer = h264.H264(client_socket, width, height, fps)
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
    # 'none'
    # 'mjpeg-30'
    # 'mjpeg-50'
    # 'mjpeg-90'
    # 'tiled-spatial'
    # 'h264'
    stream_video(compression='none')