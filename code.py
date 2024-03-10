import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define constants
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BATCH_SIZE = 32

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data
train_set = train_datagen.flow_from_directory('train',
                                              target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                              batch_size=BATCH_SIZE,
                                              class_mode='categorical')

# Load and preprocess testing data
test_set = test_datagen.flow_from_directory('archive(1)/test',
                                            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical')


# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_set.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_set,
          steps_per_epoch=train_set.samples // BATCH_SIZE,
          epochs=10,
          validation_data=test_set,
          validation_steps=test_set.samples // BATCH_SIZE)

# Evaluate the model
loss, accuracy = model.evaluate(test_set)
print(f'Test Accuracy: {accuracy}')
