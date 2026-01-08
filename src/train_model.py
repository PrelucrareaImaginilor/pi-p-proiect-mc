import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D


# Setari generale
DATA_DIR = "../data/splits"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 1
CLASSES = ["mel", "nv", "bcc", "bkl"]

# Generatoare de date
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=False
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Definirea modelului CNN
#model = Sequential([
#    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
#    MaxPooling2D(2,2),
#    Conv2D(64, (3,3), activation="relu"),
#   MaxPooling2D(2,2),
#    Flatten(),
#    Dense(128, activation="relu"),
#    Dropout(0.3),
#    Dense(len(CLASSES), activation="softmax")
#])

# Model de bază pre-antrenat
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

# Înghețăm straturile de bază
base_model.trainable = False

# Construim modelul final
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Antrenarea modelului CU class weights
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights
)

# Vizualizarea evolutiei acuratetei si pierderii
plt.figure(figsize=(12,5))

# Acuratetea
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoca')
plt.ylabel('Acuratete')
plt.title('Evolutia acurateței')
plt.legend()

# Pierderea
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.title('Evolutia pierderii')
plt.legend()

plt.tight_layout()
plt.show()

# Salvarea modelului
model.save("../model.keras")
print("Model salvat ca model.keras")
