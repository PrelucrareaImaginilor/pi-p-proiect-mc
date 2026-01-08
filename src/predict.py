import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = "../model.keras"
IMG_SIZE = (224,224)
CLASSES = ["bcc", "bkl", "mel", "nv"]

model = load_model(MODEL_PATH)
print("Model încărcat:", MODEL_PATH)

def predict_image(img_path):
    if not os.path.exists(img_path):
        print("Fisierul nu exista:", img_path)
        return

    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    class_name = CLASSES[class_idx]
    confidence = pred[0][class_idx] * 100
    print(f"Predictie: {class_name} ({confidence:.2f}%)")

def main():
    img_path = input("Introduceti calea imaginii: ")
    predict_image(img_path)

if __name__ == "__main__":
    main()
