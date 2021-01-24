import tensorflow as tf
from custom_dataset_preprocess import scale_dataset
from helpers.utils import load_custom_dataset
from sklearn.model_selection import train_test_split
from helpers.utils import plot_history

# Load custom data
X_transfer, y_transfer = load_custom_dataset("./preprocessed_dataset")
X_transfer = scale_dataset(X_transfer)
X_transfer = X_transfer.reshape(-1, 28, 28, 1)

# Split custom data into training and validation datasets
X_train_transfer, X_valid_transfer, y_train_transfer, y_valid_transfer = train_test_split(X_transfer, y_transfer, test_size=0.4, random_state=0, stratify=y_transfer)

# Load base model
model = tf.keras.models.load_model("./models/trained_CNN.h5")

# Define which layers to freeze
model.trainable = True
print("Number of layers in the base model: ", len(model.layers))

fine_tune_at = 12

for layer in model.layers[:fine_tune_at]:
  layer.trainable =  False

# Train
epochs = 30
history = model.fit(X_transfer, y_transfer,
                    validation_data=(X_valid_transfer, y_valid_transfer),
                    epochs=epochs)

# Save model
model.save("./models/transfer_model.h5")

# Plot learning curves
plot_history(history, "transfer_model")