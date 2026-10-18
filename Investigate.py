
import cv2
import os

from Predict import Predict

"""
Crop Square Region in image given onclick event at coordinate (x,y)
:param: src - source to the image
:param: x, y - onclick coordinates
"""
def classify_patch(src, x, y):
    # Patch retrieval
    image = cv2.imread(src)
    if image is None:
        raise ValueError("The uploaded image could not be read.")

    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("The selected point is outside the image.")

    patch = image[max(0, y-55):min(height, y+56), max(0, x-55):min(width, x+56)]
    patch_PIL = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    # Classification
    predicted_class = Predict(img=patch_PIL)

    # Bounding box and Class
    if predicted_class == "Normal":
        cv2.circle(image, (x,y), radius=2, color=(0,255,0), thickness=1)
        cv2.rectangle(image, (x-55, y-55), (x+55, y+55), color = (0,255,0), thickness = 2)
        cv2.putText(image, predicted_class, (x-55, y-60), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1.2, color = (0,255,0), thickness = 2)
    else:
        cv2.circle(image, (x,y), radius=2, color=(0,0,255), thickness=1)
        cv2.rectangle(image, (x-55, y-55), (x+55, y+55), color = (0,0,255), thickness = 2)
        cv2.putText(image, predicted_class, (x-55, y-60), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1.2, color = (0,0,255), thickness = 2)

    return image, predicted_class
