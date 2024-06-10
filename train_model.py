from keras import models, layers, optimizers
from keras.src.utils import image_dataset_from_directory

INPUT_IMAGE_SHAPE = (150, 150, 3)
INPUT_DIR_YELLOWS = "output/yellow"
INPUT_DIR_REDS = "output/red"


# Define the model
model_red = models.Sequential()
model_red.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_IMAGE_SHAPE))
model_red.add(layers.MaxPooling2D((2, 2)))
model_red.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_red.add(layers.MaxPooling2D((2, 2)))
model_red.add(layers.Flatten())
model_red.add(layers.Dense(64, activation='relu'))
model_red.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model_red.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

model_yellow = models.Sequential()
model_yellow.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_IMAGE_SHAPE))
model_yellow.add(layers.MaxPooling2D((2, 2)))
model_yellow.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_yellow.add(layers.MaxPooling2D((2, 2)))
model_yellow.add(layers.Flatten())
model_yellow.add(layers.Dense(64, activation='relu'))
model_yellow.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model_yellow.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# Train the model
# Resize the apples to 150x150 pixels beforehand, pad, do not crop

# Define the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Define the batch size
batch_size = 20

# Load and resize the images from the directories
train_generator_red = datagen.flow_from_directory(
    INPUT_DIR_REDS,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary')

train_generator_yellow = datagen.flow_from_directory(
    INPUT_DIR_YELLOWS,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary')

# Train the model
history_red = model_red.fit_generator(
    train_generator_red,
    steps_per_epoch=100,
    epochs=30)

history_yellow = model_yellow.fit_generator(
    train_generator_yellow,
    steps_per_epoch=100,
    epochs=30)
