import cv2
import numpy as np

WIDTH = 960
HEIGHT = 480

def calculate_edge_density(image, num_rows, num_cols):
    # Resize
    image = cv2.resize(image, (WIDTH, HEIGHT))

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detector
    edges = cv2.Canny(gray, 100, 200)

    # Get dimensions of the image
    height, width = edges.shape

    # Size of each tile
    tile_width = width // num_cols
    tile_height = height // num_rows

    # Initialize the matrix to store edge densities
    edge_density_matrix = np.zeros((num_rows, num_cols))

    # Process each tile
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate tile boundaries
            start_x, end_x = j * tile_width, min((j + 1) * tile_width, width)
            start_y, end_y = i * tile_height, min((i + 1) * tile_height, height)

            # Extract tile
            tile = edges[start_y:end_y, start_x:end_x]

            # Calculate edge density
            edge_density = np.count_nonzero(tile) / (tile.size)
            edge_density_matrix[i, j] = edge_density

    return edge_density_matrix

def calculate_corner_density(image, num_rows, num_cols):
    # Resize
    image = cv2.resize(image, (WIDTH, HEIGHT))

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Harris Corner Detector
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Threshold for an optimal value, it may vary depending on the image
    corners = cv2.dilate(corners, None)
    ret, corners = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)
    corners = np.uint8(corners)

    # Get dimensions of the image
    height, width = corners.shape

    # Size of each tile
    tile_width = width // num_cols
    tile_height = height // num_rows

    # Initialize the matrix to store corner densities
    corner_density_matrix = np.zeros((num_rows, num_cols))

    # Process each tile
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate tile boundaries
            start_x, end_x = j * tile_width, min((j + 1) * tile_width, width)
            start_y, end_y = i * tile_height, min((i + 1) * tile_height, height)

            # Extract tile
            tile = corners[start_y:end_y, start_x:end_x]

            # Calculate corner density
            corner_density = np.count_nonzero(tile) / (tile.size)
            corner_density_matrix[i, j] = corner_density

    return corner_density_matrix

def calculate_contour_density(image, num_rows, num_cols):
    # Resize
    image = cv2.resize(image, (WIDTH, HEIGHT))

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to find contours
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw contours
    contour_img = np.zeros_like(gray)

    # Draw contours
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)

    # Get dimensions of the image
    height, width = contour_img.shape

    # Size of each tile
    tile_width = width // num_cols
    tile_height = height // num_rows

    # Initialize the matrix to store contour densities
    contour_density_matrix = np.zeros((num_rows, num_cols))

    # Process each tile
    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate tile boundaries
            start_x, end_x = j * tile_width, min((j + 1) * tile_width, width)
            start_y, end_y = i * tile_height, min((i + 1) * tile_height, height)

            # Extract tile
            tile = contour_img[start_y:end_y, start_x:end_x]

            # Calculate contour density
            contour_density = np.count_nonzero(tile) / (tile.size)
            contour_density_matrix[i, j] = contour_density

    return contour_density_matrix


def label_edges(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if image is loaded correctly
    if image is None:
        print("Error: Image not found.")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detector
    edges = cv2.Canny(gray_image, 100, 200)

    # Iterate over the edges and mark them on the original image
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] != 0:  # Edge pixel found
                image[y, x] = [0, 0, 255]  # Mark with red dot

    # Save the result to a file
    cv2.imwrite(output_path, image)
    print(f"Image saved as {output_path}")

def label_corners(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if image is loaded correctly
    if image is None:
        print("Error: Image not found.")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Harris Corner Detector
    corners = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    # Dilate corner image to enhance corner points
    corners = cv2.dilate(corners, None)

    # Threshold for an optimal value, it may vary depending on the image
    threshold = 0.01 * corners.max()

    # Iterate over the corners and mark them on the original image
    for y in range(corners.shape[0]):
        for x in range(corners.shape[1]):
            if corners[y, x] > threshold:  # Corner pixel found
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Mark with red dot

    # Save the result to a file
    cv2.imwrite(output_path, image)
    print(f"Image saved as {output_path}")

def label_contours(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Check if image is loaded correctly
    if image is None:
        print("Error: Image not found.")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to find contours
    ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw contours
    contour_img = np.zeros_like(gray_image)

    # Draw contours
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)

    # Iterate over the contour image and mark them on the original image
    for y in range(contour_img.shape[0]):
        for x in range(contour_img.shape[1]):
            if contour_img[y, x] != 0:  # Contour pixel found
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)  # Mark with red dot

    # Save the result to a file
    cv2.imwrite(output_path, image)
    print(f"Image saved as {output_path}")


# image = cv2.imread('../../assets/test1.jpg')
# edge_matrix = calculate_edge_density(image, 3, 6)
# print(edge_matrix)

# image = cv2.imread('../../assets/test1.jpg')
# corner_matrix = calculate_corner_density(image, 3, 6)
# print(corner_matrix)

# image = cv2.imread('../../assets/test1.jpg')
# contour_matrix = calculate_contour_density(image, 3, 6) 
# print(contour_matrix)

# label_edges('../../assets/test1.jpg', 'edge_labeled_image.jpg')

# label_corners('../../assets/test1.jpg', 'corner_labeled_image.jpg')

# label_contours('../../assets/test1.jpg', 'contour_labeled_image.jpg')

