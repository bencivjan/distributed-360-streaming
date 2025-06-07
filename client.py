import cv2
import socket
import numpy as np
import struct
import time
from datetime import datetime
import sys
import os
from streamers import mjpeg, basic, tile_spatial, webp
from logger import Logger

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

    cap = cv2.VideoCapture('videos/ny_driving.nut')
    client_socket = socket.socket()
    client_socket.settimeout(5)  # 5 seconds timeout
    while True:
        try:
            client_socket.connect((TCP_IP, TCP_PORT))
            break
        except OSError:
            print("Unable to connect to server socket, retrying...")
            datetime_obj = datetime.fromtimestamp(time.time())
            readable_time = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
            with open("errors.out", "a") as err_file:
                err_file.write(f'{readable_time}: Unable to connect to server at {TCP_IP}\n')
            time.sleep(5)

    try:
        target_fps = 10 # Thottle to set fps for energy consistency
        # Calculate the time to wait between frames
        frame_time = 1.0 / target_fps
        frames_read = 0
        test_start_time = time.time()

        print(f'Streaming with {compression} compression')
        if compression == 'none':
            logger = Logger(f'./basic_logs.json')
            client_socket.sendall(struct.pack('B', 0x0))
            streamer = basic.Basic(client_socket, logger=logger)
        elif compression == 'mjpeg-30':
            logger = Logger(f'./mjpeg30_logs.json')
            client_socket.sendall(struct.pack('B', 0x1))
            streamer = mjpeg.Mjpeg(client_socket, qf=30, logger=logger)
        elif compression == 'mjpeg-50':
            logger = Logger(f'./mjpeg50_logs.json')
            client_socket.sendall(struct.pack('B', 0x2))
            streamer = mjpeg.Mjpeg(client_socket, qf=50, logger=logger)
        elif compression == 'mjpeg-90':
            logger = Logger(f'./mjpeg90_logs.json')
            client_socket.sendall(struct.pack('B', 0x3))
            streamer = mjpeg.Mjpeg(client_socket, qf=90, logger=logger)
        elif compression == 'webp-30':
            logger = Logger(f'./webp30_logs.json')
            client_socket.sendall(struct.pack('B', 0x4))
            streamer = webp.Webp(client_socket, qf=30, logger=logger)
        elif compression == 'webp-50':
            logger = Logger(f'./webp50_logs.json')
            client_socket.sendall(struct.pack('B', 0x5))
            streamer = webp.Webp(client_socket, qf=50, logger=logger)
        elif compression == 'webp-90':
            logger = Logger(f'./webp90_logs.json')
            client_socket.sendall(struct.pack('B', 0x6))
            streamer = webp.Webp(client_socket, qf=90, logger=logger)
        elif compression == 'tiled-spatial':
            logger = Logger(f'./tiled_logs.json')
            client_socket.sendall(struct.pack('B', 0x7))
            streamer = tile_spatial.TileSpatial(client_socket, logger=logger)
        elif compression == 'h264':
            bitrate = '1_5ghz'
            logger = Logger(f'./h264_{bitrate}_logs.json')
            client_socket.sendall(struct.pack('B', 0x8))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            streamer = h264.H264(client_socket, width, height, fps, logger=logger)
        else:
            print('Unsupported compression algorithm!')
            return

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            frames_read += 1
            if not ret:
                # print("Failed to capture frame")
                # print(f'Actual frame rate: {frames_read / (time.time() - test_start_time)}')
                _ = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print('Restarting video...')
                continue

            streamer.send_frame(frame)
            print(frame.nbytes)

            # Calculate elapsed time and sleep if necessary
            elapsed_time = time.time() - start_time
            time_to_wait = frame_time - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)
    finally:
        print(f'Actual frame rate: {frames_read / (time.time() - test_start_time)}')
        logger.flush()
        cap.release()
        client_socket.close()

if __name__ == '__main__':
    # 'none'
    # 'mjpeg-30'
    # 'mjpeg-50'
    # 'mjpeg-90'
    # 'webp'
    # 'tiled-spatial'
    # 'h264'
    # stream_video(compression='none')
    # stream_video(compression='mjpeg-30')
    # stream_video(compression='mjpeg-50')
    # stream_video(compression='mjpeg-90')
    # stream_video(compression='webp-30')
    # stream_video(compression='webp-50')
    # stream_video(compression='webp-90')
    # stream_video(compression='tiled-spatial')
    stream_video(compression='h264')