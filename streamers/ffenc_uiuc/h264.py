import time
import ffdec
import ffenc
import numpy as np
import cv2
import struct
import datetime

class H264:
    def __init__(self, sock, w=0, h=0, fps=0, logger=None):
        self.sock = sock
        self.logger = logger
        self.encoder = ffenc.ffenc(int(w), int(h), int(fps))
        self.decoder = ffdec.ffdec()
        self.buffer = b''
        self.send_frame_idx = 0
        self.recv_frame_idx = 0
        self.nbytes_received = 0

        self.encoder.change_settings(25000, int(fps))
        # print(w, h, fps)


    def send_frame(self, frame):
        # self.encoder.change_settings(5000, 31)
        try:
            out = self.encoder.process_frame(frame)
            # print(out.shape)

            print(f'Frame size: {out.shape[0]} bytes')
            start_time = time.time()

            self.sock.sendall(struct.pack('!d', start_time))
            self.sock.sendall(struct.pack('!I', out.shape[0]))
            self.sock.sendall(out.tobytes())
            end_time = time.time()

            log = {}

            log['frame'] = self.send_frame_idx
            log['client_send_start_time'] = start_time
            log['client_send_end_time'] = end_time
            log['client_send_duration'] = f'{end_time - start_time:.4f}'

            if self.logger:
                self.logger.log(log)
            self.send_frame_idx += 1
        except TimeoutError:
            print("Unable to send frame, connection timed out...")
            if self.logger:
                datetime_obj = datetime.fromtimestamp(time.time())
                readable_time = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
                self.logger.log({
                    'Connection timed out': readable_time
                })
        
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

        data = np.frombuffer(self.buffer, dtype=np.uint8)
        print(data.nbytes)
        frame = self.decoder.process_frame(data)
        # print(frame.size)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.buffer = b''

        log = {}

        network_duration = server_recv_end_time - client_send_start_time
        bandwidth = data_length / network_duration

        log['frame'] = self.recv_frame_idx
        log['frame_size'] = f'{data_length / 1000} KB'
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