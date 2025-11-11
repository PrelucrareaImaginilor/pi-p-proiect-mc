import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


DATA_DIR = "../data/splits"
IMG_SIZE = (128, 128) #dimenisunea la care dam rescale
BATCH_SIZE = 4
EPOCHS = 15
CLASSES = ["acne", "eczema", "psoriasis", "melanoma"]

#rescale la imagini
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.1,horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)


train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)), # 32 de filtre pentru trasaturi
    MaxPooling2D(2,2), #pentru eficienta reducem imaginea la jumatate
    Conv2D(64, (3,3), activation="relu"), # caracteristici complexe
    MaxPooling2D(2,2),
    Flatten(), #transforma matricea in vector
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(CLASSES), activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

model.save("../model.keras")
print("Model salvat ca model.keras")