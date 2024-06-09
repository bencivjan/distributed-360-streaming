import struct
import time
import numpy as np
import cv2

VID_HEIGHT = 1920
VID_WIDTH = 3840
VID_CHANNELS = 3
class Basic:
    def __init__(self, sock, logger=None):
        self.sock = sock
        self.logger = logger
        self.buffer = b''
        self.send_frame_idx = 0
        self.recv_frame_idx = 0
        self.nbytes_received = 0

    def send_frame(self, frame):
        frame_data = frame
        frame_data_len = frame.nbytes

        print(f'Frame size: {frame_data_len} bytes')
        start_time = time.time()

        self.sock.sendall(struct.pack('!d', start_time))
        self.sock.sendall(struct.pack('!I', frame_data_len))
        self.sock.sendall(frame_data)
        end_time = time.time()

        log = {}

        log['frame'] = self.send_frame_idx
        log['client_send_start_time'] = start_time
        log['client_send_end_time'] = end_time
        log['client_send_duration'] = f'{end_time - start_time:.4f}'

        if self.logger:
            self.logger.log(log)
        self.send_frame_idx += 1
        
    def get_frame(self):
        client_send_start_time = struct.unpack('!d', self.sock.recv(8))[0]
        data_length = struct.unpack('!I', self.sock.recv(4))[0]

        server_recv_start_time = time.time()
        
        while len(self.buffer) < data_length:
            data = self.sock.recv(min(data_length - len(self.buffer), 40960))
            if not data: # socket closed
                return None
            self.buffer += data

        self.nbytes_received += len(self.buffer)

        server_recv_end_time = time.time()
        frame = np.frombuffer(self.buffer, np.uint8).reshape(VID_HEIGHT, VID_WIDTH, VID_CHANNELS)
        self.buffer = b''

        log = {}

        network_duration = server_recv_end_time - client_send_start_time
        bandwidth = data_length / network_duration

        log['frame'] = self.recv_frame_idx
        log['client_send_start_time'] = client_send_start_time
        log['server_recv_start_time'] = server_recv_start_time
        log['server_recv_end_time'] = server_recv_end_time
        log['server_recv_duration'] = server_recv_end_time - server_recv_start_time
        log['network_duration'] = f'{network_duration * 1000:.4f} ms'
        log['bandwidth'] = f'{(bandwidth * 8) / (1000 * 1000):.4f} Mbps'

        if self.logger:
            self.logger.log(log)
        self.recv_frame_idx += 1

        return frame