from keras import models, layers, optimizers
from keras.src.utils import image_dataset_from_directory
import os

INPUT_IMAGE_SHAPE = (150, 150, 3)
INPUT_DIR_YELLOWS = "output/yellow"
INPUT_DIR_REDS = "output/red"
SEED = 42

# set keras backend to tensorflow

os.environ["KERAS_BACKEND"] = "tensorflow"

# Define the model
model_red = models.Sequential()
model_red.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_IMAGE_SHAPE))
model_red.add(layers.MaxPooling2D((2, 2)))
model_red.add(layers.Conv2D(64, (3, 3), activation="relu"))
model_red.add(layers.MaxPooling2D((2, 2)))
model_red.add(layers.Flatten())
model_red.add(layers.Dense(64, activation="relu"))
model_red.add(layers.Dense(1, activation="sigmoid"))

# Compile the model
model_red.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(), metrics=["acc"])

model_yellow = models.Sequential()
model_yellow.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_IMAGE_SHAPE, padding="SAME"))
model_yellow.add(layers.MaxPooling2D((2, 2)))
model_yellow.add(layers.Conv2D(64, (3, 3), activation="relu", padding="SAME"))
model_yellow.add(layers.MaxPooling2D((2, 2)))
model_yellow.add(layers.Flatten())
model_yellow.add(layers.Dense(64, activation="relu"))
model_yellow.add(layers.Dense(1, activation="sigmoid"))

# Compile the model
model_yellow.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(), metrics=["acc"])

# Train the model
# Resize the apples to 150x150 pixels beforehand, pad, do not crop
# Split it into test and train datasets
train_dataset_yellow, val_dataset_yellow = image_dataset_from_directory(INPUT_DIR_YELLOWS,
                                                                        labels="inferred",
                                                                        label_mode="binary",
                                                                        class_names=["correct", "incorrect"],
                                                                        validation_split=0.2,
                                                                        subset="both",
                                                                        image_size=(INPUT_IMAGE_SHAPE[0], INPUT_IMAGE_SHAPE[1]),
                                                                        batch_size=32,
                                                                        pad_to_aspect_ratio=True,
                                                                        seed=SEED)

train_dataset_red, val_dataset_red = image_dataset_from_directory(INPUT_DIR_REDS,
                                                                  labels="inferred",
                                                                  label_mode="binary",
                                                                  class_names=["correct", "incorrect"],
                                                                  validation_split=0.2,
                                                                  subset="both",
                                                                  image_size=(INPUT_IMAGE_SHAPE[0], INPUT_IMAGE_SHAPE[1]),
                                                                  batch_size=32,
                                                                  pad_to_aspect_ratio=True,
                                                                  seed=SEED)

model_yellow.fit(train_dataset_yellow, validation_data=val_dataset_yellow, epochs=2, verbose=1)
model_red.fit(train_dataset_red, validation_data=val_dataset_red, epochs=2, verbose=1)

# Save the model
model_yellow.save("models/yellow_apple_model.keras")
model_red.save("models/red_apple_model.keras")
