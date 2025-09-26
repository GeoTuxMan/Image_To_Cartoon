# cartoon.py
import cv2
import numpy as np

def image_to_cartoon(img_bgr: np.ndarray) -> np.ndarray:
    """
    img_bgr: OpenCV image (BGR uint8)
    return: cartoonified image (BGR uint8)
    """
    # 1. Smooth colors with bilateral filter several times
    num_bilateral = 7
    img_color = img_bgr.copy()
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    # 3. Edge detection (adaptive threshold or Canny)
    edges = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )

    # 4. Convert edges to color and bitwise AND
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img_color, edges_colored)

    return cartoon
