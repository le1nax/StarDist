import numpy as np
import matplotlib.pyplot as plt
import cv2

def apply_gamma_correction(image, gamma):
    # Apply gamma correction
    image_corrected = np.power(image / 255.0, gamma) * 255.0
    return np.clip(image_corrected, 0, 255).astype(np.uint8)

def create_lut():
    # Create a linear LUT that starts in the lower quarter and crosses the mean
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < 64:  # Lower quarter
            lut[i] = int(i * (256 / 64))  # Stretch lower values
        elif i < 192:  # Middle range
            lut[i] = int((i - 64) * (256 / (192 - 64)) + 64)  # Linear mapping
        else:  # Upper range
            lut[i] = 255  # Keep upper values unchanged
    return lut

def apply_lut(image, lut=create_lut()):
    return lut[image]

def plot_histogram(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    plt.figure(figsize=(12, 6))
    plt.title("Histogram after LUT Transformation")
    plt.bar(bins[:-1], hist, width=1, color='blue')
    plt.xlim([0, 255])
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()