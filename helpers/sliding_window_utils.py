import imutils
import pytesseract
from custom_dataset_preprocess import rgb2gray, shrink, scale_dataset
import matplotlib.pyplot as plt
import numpy as np

def pyramid(image, scale=1.5, minSize=(5, 5)):
    yield image
    
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        yield image
        
def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def detect_text(img):
    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=custom_config, lang="eng")
    images = []
    total_boxes = len(details['text'])
    for sequence_number in range(total_boxes):
        if int(details['conf'][sequence_number]) > 30:
            (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
            img = img[y:y+h, x:x+w]
            images.append(img)
    return images

def img_preprocessing(img):
    img = rgb2gray(img)
    print(img.shape)
    print(img.dtype)
    img = 255.0 - img
    img = shrink(img, height=28)
    img = scale_dataset(img)
    return img

# def handwriting_detection(img, model, scale=2, stepSize=28, threshold=0.9):
#     alphabet = "abcdefghijklmnopqrstuvwxyz"
#     img = img_preprocessing(img)
    
#     (winW, winH) = (28, 28)
#     for resized in pyramid(img, scale):
#         for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
#             if window.shape[0] != winH or window.shape[1] != winW:
#                 continue
#             clone = resized[y:y+winW, x:x+winH]
#             y_pred = model.predict(clone.reshape(1, 28, 28, 1))
#             y_pred_prob = y_pred.max(axis=1)
#             y_pred = y_pred.argmax(axis=1)
#             if y_pred_prob > threshold:
#                 fig, ax = plt.subplots(1)
#                 ax.imshow(clone, cmap="gray")
#                 ax.set_title(f"Predicted letter: {alphabet[y_pred[0]]}\nProb: {y_pred_prob}")
#                 plt.show()

def handwriting_detection(img, model, scale=2, stepSize=28, threshold=0.9):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    img = img_preprocessing(img)
    
    (winW, winH) = (14, 28)
    # for resized in pyramid(img, scale):
    for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        clone = np.ones((28, 28)) * (-1)
        clone[:, 7:21] = img[y:y+winH, x:x+winW]
        y_pred = model.predict(clone.reshape(1, 28, 28, 1))
        y_pred_prob = y_pred.max(axis=1)
        y_pred = y_pred.argmax(axis=1)
        if y_pred_prob > threshold:
            fig, ax = plt.subplots(1)
            ax.imshow(clone, cmap="gray")
            ax.set_title(f"Predicted letter: {alphabet[y_pred[0]]}\nProb: {y_pred_prob}")
            plt.show()