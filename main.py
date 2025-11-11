import cv2
import matplotlib.pyplot as plt

def preprocess_image(image):
    ####
    #Etapa de preprocesare prin reducere de zgomot:
    # - conversie grayscale
    # - medianBlur - smoothing/noise reduction
    ####
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #conversie de la color la grayscale
    denoised = cv2.medianBlur(grayscale,5)
    return denoised

def segment_image(image):
    ####
    # Etapa de segmentare:
    ####
    _,mask = cv2.threshold(image,0 ,255)


if __name__ == '__main__':
    img = cv2.imread("eczema.jpg") # o prima imagine de test a preprocesarii

    if img is None:
        print("Imaginea nu a putut fi deschisa.")
        exit()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_preprocessed = preprocess_image(img)

    print(f"Dimensiuni originale: {img.shape}")
    print(f"Dimensiuni dupa preprocesare: {img_preprocessed.shape}")

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Original") # afisam imaginea inainte de preprocesare
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img_preprocessed,cmap= "gray")
    plt.title("Preprocesata (Fara zgomot)")
    plt.axis("off")


    print("test")
    plt.tight_layout()
    plt.show()
