import struct

def receive_tile(client_socket):
    header = client_socket.recv(4)
    tile_data_length = struct.unpack('!I', header)[0]

    tile_data = b''
    while len(tile_data) < tile_data_length:
        chunk = client_socket.recv(min(tile_data_length - len(tile_data), 4096))
        tile_data += chunk
    
    tile = cv2.imdecode(np.frombuffer(tile_data, np.uint8), cv2.IMREAD_COLOR)
    return tile, tile_data_length

while True:
        try:
            num_rows = struct.unpack('B', client_socket.recv(1))[0]
            num_cols = struct.unpack('B', client_socket.recv(1))[0]

            # Start time (first tile sent from client)
            frame_start_time = struct.unpack('!d', client_socket.recv(8))[0]

            # tiles = [receive_tile(client_socket) for _ in range(num_rows * num_cols)]
            compressed_frame_size = 0
            tiles = [None] * (num_rows * num_cols)
            for i in range(num_rows * num_cols):
                tile, compressed_tile_size = receive_tile(client_socket)
                tiles[i] = tile
                compressed_frame_size += compressed_tile_size
            frame_size[client_ip].append(compressed_frame_size)

            combined_rows = []
            index = 0
            for i in range(num_rows):
                row_tiles = [tiles[index + j] for j in range(num_cols)]
                combined_rows.append(np.hstack(row_tiles))
                index += num_cols
            video_captures[client_ip] = np.vstack(combined_rows)