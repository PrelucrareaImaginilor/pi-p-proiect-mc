import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


MODEL_PATH = "../model.keras"
IMG_SIZE = (128, 128)
CLASSES = ["acne", "eczema", "psoriasis", "melanoma"]


model = load_model(MODEL_PATH)
print("Model încărcat:", MODEL_PATH)



def predict_image(img_path):
    if not os.path.exists(img_path):
        print("Fisierul nu exista:", img_path)
        return

    # incarcă imaginea (grayscale și redimensionare)
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")

    # Transformă imaginea în array și normalizează pixelii
    img_array = image.img_to_array(img) / 255.0

    # Adaugă batch dimension (modelul așteaptă [1,128,128,1])
    img_array = np.expand_dims(img_array, axis=0)

    # Predicție
    pred = model.predict(img_array)

    # Obține clasa cu probabilitate maximă
    class_idx = np.argmax(pred)
    class_name = CLASSES[class_idx]
    confidence = pred[0][class_idx] * 100

    print(f"Predictie: {class_name} ({confidence:.2f}%)")


# Exemplu de rulare
if __name__ == "__main__":
    img_path = input("Introduceti calea imaginii: ")
    predict_image(img_path)
