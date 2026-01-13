import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
import numpy as np

IMG_SIZE = (224, 224)
MODEL_PATH = "../model.keras"
IMG_PATH = "../data/splits/train/mel/ISIC_0024310.jpg"  # modifică imaginea

# ===========================
# 1. Încarcă imaginea
# ===========================
img = load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = img_to_array(img)
img_prepared = np.expand_dims(img_array, axis=0) / 255.0

# ===========================
# 2. Încărcare model
# ===========================
model = load_model(MODEL_PATH)

# accesăm primul strat Conv2D din MobileNetV2
base_model = model.layers[0]  # MobileNetV2
conv_layers = [layer for layer in base_model.layers if "conv" in layer.name.lower()]

if not conv_layers:
    raise ValueError("Nu am găsit straturi Conv2D în MobileNetV2!")

first_conv_layer = conv_layers[0]
print("Primul strat Conv2D:", first_conv_layer.name)

# model pentru activări
activation_model = Model(inputs=base_model.input, outputs=first_conv_layer.output)

# ===========================
# 3. Generăm activările
# ===========================
activations = activation_model.predict(img_prepared)

# activările au forma (1, H, W, num_filters)
num_filters = activations.shape[-1]

# ===========================
# 4. Afișăm primele 6 activări
# ===========================
plt.figure(figsize=(15, 10))

for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    plt.imshow(activations[0, :, :, i], cmap='viridis')
    plt.title(f"Activare {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()
