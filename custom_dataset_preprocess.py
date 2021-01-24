import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pickle

from sklearn.preprocessing import MinMaxScaler

def image_centering(img):
    left, right, top, bottom = 0, 0, 0, 0
    # print(img.shape)
    for i in range(img.shape[0]):
        if img[:, i].sum() > 0:
            left = i
            break
    for i in reversed(range(img.shape[0])):
        if img[:, i].sum() > 0:
            right = i
            break
    for i in range(img.shape[1]):
        if img[i, :].sum() > 0:
            top = i
            break
    for i in reversed(range(img.shape[1])):
        if img[i, :].sum() > 0:
            bottom = i
            break
    return img[top:bottom, left:right]

def image_padding(img):
    centered_image = np.zeros((512, 512))
    img_h, img_w = img.shape
    left = (512 - img_w) // 2
    right = 512 - left
    top = (512 - img_h) // 2
    bottom = 512 - top
    centered_image[top:top+img_h, left:left+img_w] = img
    return centered_image

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.ceil(gray).astype(np.int)

def shrink(img, height=28):
    img = imutils.resize(img, height=height)
    img = np.clip(img*2, 0, 255)
    return img.astype(np.int)

def scale_dataset(X):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(X)

if __name__ == "__main__":
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    nums = [i for i in range(1, 11)]

    preprocessed_dataset = []

    for letter in alphabet:
        for num in nums:
            img_path = "./dataset/" + letter + "/" + letter + str(num) + ".png"
            img = cv2.imread(img_path)
            img = rgb2gray(img)
            img = 255 - img

            img = image_centering(img)
            img = image_padding(img)
            save_path = "./preprocessed_dataset/centered/" + letter + "/" + letter + str(num) + "_centered.png"
            print(save_path)
            cv2.imwrite(save_path, img)

            img = shrink(img, height=28)
            save_path = "./preprocessed_dataset/small/" + letter + "/" + letter + str(num) + "_small.png"
            print(save_path)
            cv2.imwrite(save_path, img)

            preprocessed_dataset.append(img.flatten())

    preprocessed_dataset = np.array(preprocessed_dataset)
    print(preprocessed_dataset.shape)
    with open("./preprocessed_dataset/preprocessed_dataset.pickle", "wb") as f:
        pickle.dump(preprocessed_dataset, f)

    labels = np.arange(26)
    labels = np.repeat(labels, 10)
    with open("./preprocessed_dataset/preprocessed_labels.pickle", "wb") as f:
        pickle.dump(labels, f)

        
    with open("./preprocessed_dataset/preprocessed_dataset.pickle", "rb") as f:
        X = pickle.load(f)
    with open("./preprocessed_dataset/preprocessed_labels.pickle", "rb") as f:
        y = pickle.load(f)

    img = cv2.imread("./dataset/n/n9.png")
    img_preprocessed = X[138].reshape(28, 28)

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("before")
    plt.subplot(1, 2, 2)
    plt.title("after")
    plt.imshow(img_preprocessed, cmap="gray")
    plt.show()

    