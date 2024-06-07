import struct
import logger

class Basic:
    def __init__(self, sock, logger=None):
        self.sock = sock
        self.logger = logger

    def send_frame(self, frame):
        frame_data = frame
        frame_data_len = frame.nbytes

        print(f'Frame size: {frame_data_len} bytes')
        self.sock.sendall(struct.pack('!I', frame_data_len))
        self.sock.sendall(frame_data)
        
    def get_frame(self):
        data_length = struct.unpack('!I', self.sock.recv(4))[0]
        
        while len(self.buffer) < data_length:
            data = self.sock.recv(min(data_length - len(self.buffer), 40960))
            if not data: # socket closed
                return None
            self.buffer += data
        frame = self.buffer
        self.buffer = b''
        return frame