## CLASIFICAREA BOLILOR DERMATOLOGICE

| Autor / An | Titlu | Tehnologii utilizate | Metodologie / Abordare | Limitări | Rezultate | Comentarii |
|-------------|--------|----------------------|--------------------------|-----------|------------|-------------|
| **A. Akram, 2023** | Segmentation and classification of skin lesions using hybrid deep learning | U-Net / encoder-decoder pentru segmentare + CNN; OpenCV, Keras/TensorFlow | Segmentare pentru izolare leziune; clasificare CNN pe regiunea segmentată; augmentări și validare cross-dataset | Segmentarea poate eșua dacă imaginea nu este perfectă | Rezultate mai bune datorită folosirii segmentării înaintea clasificării | U-Net ajută la restituirea imaginii originale, combinând caracteristici low-level și high-level |
| **Sri Lanka Institute of Information Technology, 2020** | CURETO: Skin Diseases Detection Using Image Processing And CNN | Preprocesare imagini; CNN; interfață mobil | Achiziție imagine; preprocesare; clasificare CNN | Validare clinică; defecte în imagini | Skin sensitivity = 95.7%<br>Acne density = 85.1% | Cel mai apropiat de proiectul nostru |
| **W. Gouda, 2022** | Detection of Skin Cancer Based on Skin Lesion Images Using Deep Learning (ISIC2018 / ESRGAN preprocessing) | CNN; PyTorch/TensorFlow | Preprocesare, antrenare CNN și evaluare pe ISIC2018 | Necesită validare clinică | Rezultate îmbunătățite datorită ESRGAN | ESRGAN – tehnologie AI pentru îmbunătățirea rezoluției imaginilor |
| **Shaden Abdulaziz AlDera, Mohamed Tahar Ben Othman, 2022** | A Model for Classification and Diagnosis of Skin Disease using Machine Learning and Image Processing Techniques | Python (Spyder/Anaconda), OpenCV, Scikit-Image, Scikit-learn, NumPy, Matplotlib<br>Algoritmi: SVM (90.7%), Random Forest, K-NN | Achiziție: 377 imagini<br>Preprocesare: redimensionare, filtrare, normalizare<br>Segmentare și extragere caracteristici<br>Clasificare ML | Set de date mic și puțin divers | SVM – cea mai mare acuratețe; RF – cea mai bună pt. acnee; K-NN – 67.1% medie | Lipsesc detalii despre tonuri de piele, aparat foto, iluminare |
| **Haijing Wu, 2020** | A deep learning, image-based approach for automated diagnosis for inflammatory skin diseases | Model deep learning EfficientNet-b4; imagini clinice etichetate | Proces end-to-end: prelucrare, antrenare, clasificare în HC / Pso / Ecz-AD | Se concentrează doar pe boli inflamatorii; posibilă dependență de dataset | Acuratețe generală 95.80% ± 0.09%<br>Pso 89.46%, Ecz/AD 92.57% | Model performant, dar necesită validare pe seturi mai diverse |
| **K. A. Muhaba et al., 2021** | Automatic skin disease diagnosis using deep learning from clinical image and patient information | Model deep learning MobileNet V2 pre-antrenat; imagini + date clinice | Achiziție imagini (286 pacienți) + metadate clinice; augmentare; antrenare MobileNet V2 pentru clasificare a 5 boli cutanate | Date dintr-un singur set; lipsă diversitate; dependență de meta-date | Acuratețe 97.5%; Sensibilitate 97.7%; Precizie 97.7% | Util în medii cu resurse reduse; combinația imagini + date clinice e foarte eficientă |


![Schema Bloc](images/SchemaPI.drawio.png)

##  Etape ale proiectului

1. **Încărcare imagini**  
   - Alegem un dataset popular cu cât mai multe imagini (ex. *ISIC*, *HAM10000*).  
   - Verificăm extensia fișierelor și formatăm datele pentru procesare ulterioară.

2. **Preprocesare**  
   - Trecem prin pașii de pregătire a imaginilor: redimensionare la dimensiunea de intrare a modelului (ex. 224×224), normalizare a pixelilor și augmentare (rotație, flip, zoom) pentru a mări setul de antrenare.

3. **Model CNN**  
   - Conține modelul de învățare profundă pentru clasificare.  
   - Putem folosi un model pre-antrenat (ex. *ResNet*, *MobileNetV2*, *EfficientNet*) și îl reantrenăm pe date dermatologice (transfer learning).

4. **Clasificare și evaluare**  
   - Pe baza ieșirii modelului CNN, clasificăm imaginea într-o categorie (ex. *benign*, *malign*, *acnee*, *eczemă* etc.).  
   - Calculăm metrici precum acuratețe, precizie, recall și F1-score.

5. **Interfață / Rezultate**  
   - Prezentăm rezultatul într-un format ușor de interpretat — de exemplu, un script Python care afișează imaginea și predicția.
