import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Setari generale
DATA_DIR = "../data/splits"
IMG_SIZE = (128, 128)  # dimensiunea la care redimensionăm imaginile
BATCH_SIZE = 4
EPOCHS = 15
CLASSES = ["acne", "eczema", "psoriasis", "melanoma"]

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

# Definirea modelului CNN
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Antrenarea modelului și salvarea istoricului
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

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
