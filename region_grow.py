import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def loadImage(file):
    """
    Load an image from a file, convert to grayscale, and resize.

    Args:
        file (str): Path to the image file.

    Returns:
        numpy.ndarray: Resized grayscale image.
    """
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
    return img

def median(img, blur):
    """
    Apply median blur to the image.

    Args:
        img (numpy.ndarray): Input image.
        blur (int): Kernel size for the median blur.

    Returns:
        numpy.ndarray: Blurred image.
    """
    img_median = cv2.medianBlur(img, blur)
    return img_median

def regionGrowing(img, seed_point, thresh):
    """
    Perform region growing algorithm on the image.

    Args:
        img (numpy.ndarray): Input image.
        seed_point (tuple): Seed point for region growing.
        thresh (int): Threshold value for region growing.

    Returns:
        numpy.ndarray: Binary mask of the grown region.
    """
    h, w = img.shape
    seed_value = img[seed_point]
    mask = np.zeros((h, w), np.uint8)
    mask[seed_point] = 255
    
    queue = [seed_point]
    while len(queue) > 0:
        x, y = queue.pop(0)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] == 0:
                if abs(int(img[nx, ny]) - int(seed_value)) < thresh:
                    mask[nx, ny] = 255
                    queue.append((nx, ny))
    
    return mask

def contourArea(mask):
    """
    Find contours in the mask and calculate the area of the largest contour.

    Args:
        mask (numpy.ndarray): Input binary mask.

    Returns:
        tuple: Area of the largest contour and list of all contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
    else:
        area = 0
        contours = []
    return area, contours

def drawContour(img, contours):
    """
    Draw contours on the image.

    Args:
        img (numpy.ndarray): Input binary image.
        contours (list): List of contours to be drawn.

    Returns:
        numpy.ndarray: Image with contours drawn.
    """
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_color, contours, -1, (0, 0, 255), 2)
    return img_color

def processImages(image_files, output_folder):
    """
    Process a list of images: apply blurring, region growing, and find contour areas.

    Args:
        image_files (list): List of paths to image files.
        output_folder (str): Folder to save processed images.

    Returns:
        list: Areas of the largest contours in the images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    areas = []
    processed_images = []

    for i, file in enumerate(image_files):
        img = loadImage(file)
        img = median(img, 23)

        h, w = img.shape
        seed_point = (h // 2, w // 2)
        thresh = 7

        mask = regionGrowing(img, seed_point, thresh)
        area, contours = contourArea(mask)
        areas.append(area)

        img_color = drawContour(mask, contours)
        
        output_file = os.path.join(output_folder, f"processed_{i}.png")
        cv2.imwrite(output_file, img_color)
        processed_images.append(output_file)
    
    createGif(processed_images, os.path.join(output_folder, "output.gif"))
    return areas

def createGif(image_files, output_file):
    """
    Create a GIF from a list of image files.

    Args:
        image_files (list): List of paths to image files.
        output_file (str): Path to save the output GIF file.
    """
    images = [Image.open(img) for img in image_files]
    images[0].save(output_file, save_all=True, append_images=images[1:], loop=0, duration=500)

image_folder1 = 'images/Images_M2_3ml'
output_folder1 = 'output/Output_M2_3ml'

image_folder2 = 'images/Images_M2_5ml'  
output_folder2 = 'output/Output_M2_5ml'

# Load all images from the folder
image_files1 = [os.path.join(image_folder1, f) for f in os.listdir(image_folder1) if f.endswith('.tif')]
image_files2 = [os.path.join(image_folder2, f) for f in os.listdir(image_folder2) if f.endswith('.tif')]

areas1 = processImages(image_files1, output_folder1)
print("Areas:", areas1)

areas2 = processImages(image_files2, output_folder2)
print("Areas:", areas2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(areas1, 'r-', label='Set - 3 ml')
ax1.set_xlabel('Image Number', fontsize=14)
ax1.set_ylabel('Chip Area (px$^2$)', fontsize=14)
ax1.legend()

ax2.plot(areas2, 'r-', label='Set - 5 ml')
ax2.set_xlabel('Image Number', fontsize=14)
ax2.set_ylabel('')
ax2.legend()

fig.suptitle('Segmentation – Region Growing')
plt.savefig('Method2.png')
plt.show()
