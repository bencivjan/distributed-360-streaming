import struct
import cv2
import numpy as np
import time
from feature import calculate_compression_profile

class TileSpatial:
    def __init__(self, sock, logger=None):
        self.sock = sock
        self.logger = logger
        self.send_frame_idx = 0
        self.recv_frame_idx = 0

    @staticmethod
    def cap_compression_profile(matrix):
        transformed_matrix = matrix * 100 * 15
        transformed_matrix = np.clip(transformed_matrix, 1, 50)
        transformed_matrix = transformed_matrix.astype(int)
        return transformed_matrix

    def send_tile(self, tile, quality):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        _, tile_encoded = cv2.imencode('.jpg', tile, encode_param)
        tile_data = tile_encoded.tobytes()
        header = struct.pack('!I', len(tile_data))
        self.sock.sendall(header)
        self.sock.sendall(tile_data)

    def send_image(self, image, qualities):
        rows = len(qualities)
        cols = len(qualities[0])
        h, w, _ = image.shape
        tile_height, tile_width = h // rows, w // cols
        for i in range(rows):
            for j in range(cols):
                tile = image[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]
                self.send_tile(tile, qualities[i][j])

    def send_frame(self, frame):
        # qualities = [[100, 100, 100, 100], [100, 100, 100, 100]]
        qualities = self.cap_compression_profile(calculate_compression_profile(frame, 2, 4))
        
        print("Compression Profile:\n", qualities)
        # Send the number of rows and columns
        num_rows, num_cols = len(qualities), len(qualities[0])

        # Send the timestamp
        start_time = time.time()
        self.sock.sendall(struct.pack('!d', start_time))
        
        self.sock.sendall(struct.pack('B', num_rows))
        self.sock.sendall(struct.pack('B', num_cols))
        
        self.send_image(frame, qualities)
        end_time = time.time()

        log = {}

        log['frame'] = self.send_frame_idx
        log['client_send_start_time'] = start_time
        log['client_send_end_time'] = end_time
        log['client_send_duration'] = f'{end_time - start_time:.4f}'

        if self.logger:
            self.logger.log(log)
        self.send_frame_idx += 1

    def receive_tile(self):
        header = self.sock.recv(4)
        tile_data_length = struct.unpack('!I', header)[0]

        tile_data = b''
        while len(tile_data) < tile_data_length:
            chunk = self.sock.recv(min(tile_data_length - len(tile_data), 4096))
            tile_data += chunk
        
        self.frame_data_length += tile_data_length
        tile = cv2.imdecode(np.frombuffer(tile_data, np.uint8), cv2.IMREAD_COLOR)
        return tile

    def get_frame(self):
        client_send_start_time = struct.unpack('!d', self.sock.recv(8))[0]
        
        server_recv_start_time = time.time()

        num_rows = struct.unpack('B', self.sock.recv(1))[0]
        num_cols = struct.unpack('B', self.sock.recv(1))[0]

        self.frame_data_length = 0
        tiles = [self.receive_tile() for _ in range(num_rows * num_cols)]
        server_recv_end_time = time.time()

        combined_rows = []
        index = 0
        for i in range(num_rows):
            row_tiles = [tiles[index + j] for j in range(num_cols)]
            combined_rows.append(np.hstack(row_tiles))
            index += num_cols
        frame = np.vstack(combined_rows)

        log = {}

        network_duration = server_recv_end_time - client_send_start_time
        bandwidth = self.frame_data_length / network_duration

        log['frame'] = self.recv_frame_idx
        log['client_start_time'] = client_send_start_time
        log['server_recv_start_time'] = server_recv_start_time
        log['server_recv_end_time'] = server_recv_end_time
        log['server_recv_duration'] = server_recv_end_time - server_recv_start_time
        log['network_duration'] = f'{network_duration * 1000:.4f} ms'
        log['bandwidth'] = f'{(bandwidth * 8) / (1000 * 1000):.4f} Mbps'

        if self.logger:
            self.logger.log(log)
        self.recv_frame_idx += 1

        return frame