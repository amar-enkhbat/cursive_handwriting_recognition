# Cursive Handwriting Recognition (Incomplete)
This repository is for training a cursive handwriting model for Image and Video Recognition course final project at TITech 2020 4Q.  
This model was trained on Kaggle A-Z dataset and further transfer learned using custom cursive dataset.  
Even though this model can recognize individual letters and can detect cursive letters using sliding window algorithm, this model __DOES NOT WORK__ on cursive handwritten words.

## Getting started
Edit the image filename in the 
```
handwriting_recognition.py
```
file and run. This file uses pretrained model.
If you want to train using on your own dataset edit files:
```
train_base_CNN_model.py
train_transfer_model.py
```

## Prerequisites
```
tensorflow 2.4
pytessaract
opencv2
```

