from streamers.mjpeg import Mjpeg
import cv2
import time
import struct

# Inherit from Mjpeg because all functions are identical except send_frame
class Webp(Mjpeg):
    def send_frame(self, frame):
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), self.qf]
        _, frame_encoded = cv2.imencode('.webp', frame, encode_param)
        frame_data = frame_encoded.tobytes()
        frame_data_len = len(frame_data)

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