import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
def load_kaggle_A_Z(path):
    kaggle_A_Z = pd.read_csv(path)
    dataset = kaggle_A_Z.iloc[:, 1:].values
    labels = kaggle_A_Z.iloc[:, 0].values
    return dataset, labels

def load_custom_dataset(path):
    with open(path + "/preprocessed_dataset.pickle", "rb") as f:
        dataset = pickle.load(f)
    with open(path + "/preprocessed_labels.pickle", "rb") as f:
        labels = pickle.load(f)
    return dataset, labels

def plot_correct_incorrect(dataset, labels, model):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    pred_labels = model.predict(dataset).argmax(axis=1)
    
    correct_idc = np.argwhere(pred_labels == labels).flatten()
    incorrect_idc = np.argwhere(pred_labels != labels).flatten()
    print("# of correctly predicted samples:", len(correct_idc))
    print("# of incorrectly predicted samples:", len(incorrect_idc))

    plt.figure(figsize=(10, 10))
    plt.title("Correctly predicted samples")
    for i in range(1, 6):
        plt.subplot(1, 5, i)
        true_label = alphabet[labels[correct_idc[i]]]
        pred_label = alphabet[pred_labels[correct_idc[i]]]
        plt.title(f"True Label: {true_label}\n Predicted Label: {pred_label}")
        plt.imshow(dataset[correct_idc[i]], cmap="gray")
        
    
    if len(incorrect_idc) < 5:
        plt.figure(figsize=(10, 10))
        plt.title("Incorrectly predicted samples")
        for i in range(1, len(incorrect_idc) + 1):
            plt.subplot(1, 5, i)
            true_label = alphabet[labels[incorrect_idc[i-1]]]
            pred_label = alphabet[pred_labels[incorrect_idc[i-1]]]
            plt.title(f"True Label: {true_label}\n Predicted Label: {pred_label}")
            plt.imshow(dataset[incorrect_idc[i-1]], cmap="gray")
    else:
        plt.figure(figsize=(10, 10))
        plt.title("Incorrectly predicted samples")
        for i in range(1, 6):
            plt.subplot(1, 5, i)
            true_label = alphabet[labels[incorrect_idc[i-1]]]
            pred_label = alphabet[pred_labels[incorrect_idc[i-1]]]
            plt.title(f"True Label: {true_label}\n Predicted Label: {pred_label}")
            plt.imshow(dataset[incorrect_idc[i-1]], cmap="gray")

def plot_history(history, model_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    
    
    plt.savefig("./results/" + model_name + "/learning_curve.png")
    plt.show()