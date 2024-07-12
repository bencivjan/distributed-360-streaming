import subprocess
import os
import sys
import cv2
import time
import json
import socket
from datetime import datetime
import struct
from collections import defaultdict

mod_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'streamers', 'ffenc_uiuc'))
if mod_dir not in sys.path:
    sys.path.append(mod_dir)
from streamers.ffenc_uiuc import ffenc

# Scaling governor must be set to userspace
# `sh -c 'sudo cpufreq-set -g userspace'`
def set_cpu_freq(cpu_freq):
    with open('/sys/devices/system/cpu/cpufreq/policy0/scaling_governor', 'r') as file:
        assert file.read().strip() == 'userspace', 'Scaling governor must be set to userspace\n`sudo cpufreq-set -g userspace`'
    
    cpu_freq = str(cpu_freq)
    print(f'Setting cpu freqency to {cpu_freq} KHz')

    command = f"echo {cpu_freq} | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)

    if result.returncode == 0:
        print("Successfully set cpu frequency")
    else:
        print("Failed to set cpu frequency")
        print(f"Error: {result.stderr}")

def profile(sock=None, cycles=1, replay_forever=False):
    cap = cv2.VideoCapture('videos/climbing.nut')

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    encoder = ffenc.ffenc(width, height, fps)

    # FREQS = [15 * 10**5, 16 * 10**5, 17 * 10**5, 18 * 10**5, 19 * 10**5, 20 * 10**5,
    #          21 * 10**5, 22 * 10**5, 23 * 10**5, 24 * 10**5] # In KHz
    FREQS=[24 * 10**5]

    test_results = defaultdict(int)
    test_results_fps = defaultdict(int)

    for freq in FREQS:
        set_cpu_freq(freq)

        print(f'Timing h264 compression at {str(freq / 1_000_000)} GHz')
        print(f'Running {cycles} cycle(s)')

        test_start_time = time.time()
        total_frames = 0

        for cycle in range(cycles):
            print(f'Starting cycle {cycle+1}')
            if replay_forever:
                print('Streaming video continuously...')
            total_time = 0
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        # total_time = time.time() - test_start_time
                        # test_results[f'{freq / 1_000_000}'] += total_time
                        
                        print('Restarting video...')
                        _ = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if replay_forever:
                            continue
                        else:
                            break

                    total_frames += 1
                    
                    out = encoder.process_frame(frame)
                    if sock:
                        start_time = time.time()
                        sock.sendall(struct.pack('!d', start_time))
                        sock.sendall(struct.pack('!I', out.shape[0]))
                        sock.sendall(out.tobytes())
                except:
                    break

            print('Finished profiling')
            print(f'{freq / 1_000_000} GHz: {total_frames} frames')
            print(f'{freq / 1_000_000} GHz: {total_time} seconds')
        
        total_time = time.time() - test_start_time
        test_results[f'{freq / 1_000_000}'] = total_time / cycles
        test_results_fps[f'{freq / 1_000_000} fps'] = total_frames / total_time

    cap.release()

    out_file = 'h264_cpufreq_profile.json'

    if not os.path.exists(out_file):
        open(out_file, 'w').close()

    with open(out_file, 'r+') as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = []
        history.append(test_results)
        history.append(test_results_fps)
        f.seek(0)
        json.dump(history, f, indent=4)
        f.truncate()

    print(f'Saved logs to {out_file}\n')

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        TCP_IP = sys.argv[1]
        TCP_PORT = 8010

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

    else:
        print('No server IP provided, profiling compression only...')
        client_socket = None

    # profile(cycles=10)

    if client_socket:
        client_socket.sendall(struct.pack('B', 0x8)) # Tell server we are streaming with h264
        profile(sock=client_socket, replay_forever=True)