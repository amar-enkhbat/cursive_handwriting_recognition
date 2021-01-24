import tensorflow as tf
import cv2
from helpers.sliding_window_utils import handwriting_detection

model = tf.keras.models.load_model("./models/transfer_model.h5")

img = cv2.imread("./sample_images/hello.png")
handwriting_detection(img, model, scale=2, stepSize=10, threshold=0.9)