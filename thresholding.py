import cv2
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

def histogram(img):
    """
    Calculate the histogram of the image and find the bin with the minimum value within the first 20 bins.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        int: The bin index with the minimum value within the first 20 bins.
    """
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist = hist[0:20].argmin()
    return hist

def threshold(img, threshold):
    """
    Apply binary inverse thresholding to the image.

    Args:
        img (numpy.ndarray): Input image.
        threshold (int): Threshold value.

    Returns:
        numpy.ndarray: Binary image after thresholding.
    """
    _, binary_inv = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary_inv

def contourArea(img):
    """
    Find contours in the image and calculate the area of the largest contour.

    Args:
        img (numpy.ndarray): Input binary image.

    Returns:
        tuple: Area of the largest contour and image with contours drawn.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)
    else:
        area = 0
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return area, img_color

def processImages(image_files, output_folder):
    """
    Process a list of images: apply blurring, calculate histogram, thresholding, and find contour areas.

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
        hist = histogram(img)
        binary = threshold(img, hist)
        area, img_color = contourArea(binary)
        areas.append(area)
        
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

image_folder1 = 'images/Images_M1_3ml'
output_folder1 = 'output/Output_M1_3ml'

image_folder2 = 'images/Images_M1_5ml'  
output_folder2 = 'output/Output_M1_5ml'

image_files1 = [os.path.join(image_folder1, f) for f in os.listdir(image_folder1) if f.endswith('.tif')]
image_files2 = [os.path.join(image_folder2, f) for f in os.listdir(image_folder2) if f.endswith('.tif')]
areas1 = processImages(image_files1, output_folder1)
print("Areas:", areas1)

areas2 = processImages(image_files2, output_folder2)
print("Areas:", areas2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(areas1, 'g-', label='Set - 3 ml')
ax1.set_xlabel('Image Number', fontsize=14)
ax1.set_ylabel('Chip Area (px$^2$)', fontsize=14)
ax1.legend()

ax2.plot(areas2, 'g-', label='Set - 5 ml')
ax2.set_xlabel('Image Number', fontsize=14)
ax2.set_ylabel('')
ax2.legend()

fig.suptitle('Segmentation – Thresholding')
plt.savefig('Method1.png')
plt.show()
