import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split
from helpers.utils import load_kaggle_A_Z, plot_history
from custom_dataset_preprocess import scale_dataset

# Load Kaggle A-Z dataset
X, y = load_kaggle_A_Z("./kaggle_A-Z_dataset/A_Z Handwritten Data.csv")
X = scale_dataset(X)
X = X.reshape(-1, 28, 28, 1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = "relu", input_shape = (28,28,1), kernel_regularizer=regularizers.l2(0.0005)),
    layers.Conv2D(filters = 32, kernel_size = 5, strides = 1, use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size = 2, strides = 2),
    layers.Dropout(0.25),
    layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = "relu", kernel_regularizer=regularizers.l2(0.0005)),
    layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size = 2, strides = 2),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(units = 256, use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(units = 128, use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dense(units = 84, use_bias=False),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.25),
    layers.Dense(units = 26, activation = "softmax"),
    ],
    name="Base_CNN_Model")

checkpoint_path = "model_checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Compile Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=30, 
                    validation_data=(X_valid, y_valid),
                    callbacks=[cp_callback])

# Save trained model
model.save("./models/trained_CNN.h5")

# Plot history
plot_history(history, "base_CNN_model")