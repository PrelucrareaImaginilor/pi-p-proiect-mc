import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Calea către modelul deja antrenat
MODEL_PATH = "../model.keras"
IMG_SIZE = (128, 128)
CLASSES = ["acne", "eczema", "psoriasis", "melanoma"]

# Încarcă modelul o singură dată
model = load_model(MODEL_PATH)
print("Model încărcat:", MODEL_PATH)

def predict_image(img_path):
    if not os.path.exists(img_path):
        print("Fișierul nu există:", img_path)
        return

    # Încarcă imaginea (grayscale și resize)
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicție
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    class_name = CLASSES[class_idx]
    confidence = pred[0][class_idx] * 100

    print(f"\nPredicție: {class_name} ({confidence:.2f}%)\n")

if __name__ == "__main__":
    print("Verificare model - Introduceți calea unei imagini sau 'exit' pentru a închide.")
    while True:
        img_path = input("Calea imaginii: ").strip()
        if img_path.lower() in ["exit", "quit"]:
            print("Ieșire din program.")
            break
        predict_image(img_path)
