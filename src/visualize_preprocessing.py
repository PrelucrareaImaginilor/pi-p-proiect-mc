import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import Input
import os


MODEL_PATH = "../model.keras"
IMG_SIZE = (128, 128)
CLASSES = ["acne", "eczema", "psoriasis", "melanoma"]


img_path = "../data/sample_images/acne/acne2.jpg"
if not os.path.exists(img_path):
    print("Imaginea nu exista:", img_path)
    exit()


model = load_model(MODEL_PATH)
print("Model încarcat:", MODEL_PATH)


orig_img = image.load_img(img_path)
preproc_img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
preproc_array = image.img_to_array(preproc_img) / 255.0
img_array_batch = np.expand_dims(preproc_array, axis=0)  # [1,128,128,1]


input_tensor = Input(shape=(128,128,1))
first_conv_output = model.layers[0](input_tensor)
activation_model = Model(inputs=input_tensor, outputs=first_conv_output)


activations = activation_model.predict(img_array_batch)
first_layer_activation = activations  # [1, H, W, num_filters]


plt.figure(figsize=(16,6))

# 1. Imagine originala
plt.subplot(2,4,1)
plt.imshow(orig_img)
plt.title("Originala")
plt.axis("off")

# 2. Imagine preprocesata (grayscale)
plt.subplot(2,4,2)
plt.imshow(preproc_array.squeeze(), cmap="gray")
plt.title("Preprocesata")
plt.axis("off")

# 3-5. Primele 3 feature maps
num_filters_to_show = min(3, first_layer_activation.shape[-1])
for i in range(num_filters_to_show):
    plt.subplot(2,4,i+3)
    plt.imshow(first_layer_activation[0,:,:,i], cmap='viridis')
    plt.title(f"Filter {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
