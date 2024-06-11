from keras import models, layers, optimizers
from keras.src.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
import os

INPUT_IMAGE_SHAPE = (50, 50, 3)
INPUT_DIR_YELLOWS = "output/yellow"
INPUT_DIR_REDS = "output/red"
SEED = 42

# set keras backend to tensorflow

os.environ["KERAS_BACKEND"] = "tensorflow"

# Define the model
model_red = models.Sequential()
model_red.add(layers.BatchNormalization(input_shape=INPUT_IMAGE_SHAPE))
model_red.add(layers.Conv2D(32, (3, 3), activation="relu", padding="SAME", kernel_initializer="he_normal"))
model_red.add(layers.MaxPooling2D((2, 2)))
model_red.add(layers.Conv2D(64, (3, 3), activation="relu", padding="SAME", kernel_initializer="he_normal"))
model_red.add(layers.MaxPooling2D((2, 2)))
model_red.add(layers.Flatten())
model_red.add(layers.Dense(64, activation="relu"))
model_red.add(layers.Dense(1, activation="sigmoid"))

# Compile the model
model_red.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(learning_rate=0.001), metrics=["acc"])


model_yellow = models.Sequential()
model_yellow.add(layers.BatchNormalization(input_shape=INPUT_IMAGE_SHAPE))
model_yellow.add(layers.Conv2D(32, (3, 3), activation="relu", padding="SAME", kernel_initializer="he_normal"))
model_yellow.add(layers.MaxPooling2D((2, 2)))
model_yellow.add(layers.Conv2D(64, (3, 3), activation="relu", padding="SAME", kernel_initializer="he_normal"))
model_yellow.add(layers.MaxPooling2D((2, 2)))
model_yellow.add(layers.Flatten())
model_yellow.add(layers.Dense(64, activation="relu"))
model_yellow.add(layers.Dense(1, activation="sigmoid"))

# Compile the model
model_yellow.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(learning_rate=0.001), metrics=["acc"])

# Train the model
train_dataset_yellow, val_dataset_yellow = image_dataset_from_directory(INPUT_DIR_YELLOWS,
                                                                        labels="inferred",
                                                                        label_mode="binary",
                                                                        class_names=["incorrect", "correct"],
                                                                        validation_split=0.2,
                                                                        subset="both",
                                                                        image_size=(INPUT_IMAGE_SHAPE[0], INPUT_IMAGE_SHAPE[1]),
                                                                        batch_size=32,
                                                                        pad_to_aspect_ratio=True,
                                                                        seed=SEED)

train_dataset_red, val_dataset_red = image_dataset_from_directory(INPUT_DIR_REDS,
                                                                  labels="inferred",
                                                                  label_mode="binary",
                                                                  class_names=["incorrect", "correct"],
                                                                  validation_split=0.2,
                                                                  subset="both",
                                                                  image_size=(INPUT_IMAGE_SHAPE[0], INPUT_IMAGE_SHAPE[1]),
                                                                  batch_size=32,
                                                                  pad_to_aspect_ratio=True,
                                                                  seed=SEED)

history_yellow = model_yellow.fit(train_dataset_yellow, validation_data=val_dataset_yellow, epochs=30, verbose=1)
history_red = model_red.fit(train_dataset_red, validation_data=val_dataset_red, epochs=30, verbose=1)

# Save the model
model_yellow.save("models/yellow_apple_model.keras")
model_red.save("models/red_apple_model.keras")

# Plot the training history
plt.plot(history_yellow.history["acc"], label="Training accuracy")
plt.plot(history_yellow.history["val_acc"], label="Validation accuracy")
plt.title("Yellow apple model")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history_red.history["acc"], label="Training accuracy")
plt.plot(history_red.history["val_acc"], label="Validation accuracy")
plt.title("Red apple model")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
