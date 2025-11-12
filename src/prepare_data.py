import os
import shutil
from sklearn.model_selection import train_test_split


IMAGES_DIR="../data/sample_images"
OUTPUT_DIR="../data/splits"

CLASSES=["acne","eczema","psoriasis","melanoma"]

def main():
    images = []
    for cls in CLASSES:
        cls_path = os.path.join(IMAGES_DIR, cls)
        if os.path.exists(cls_path):
            for img_file in os.listdir(cls_path):
                if img_file.lower().endswith(".jpg"):
                    images.append({
                        "image_path": os.path.join(cls_path, img_file),
                        "dx": cls
                    })

    print("Total imagini gasite:", len(images))
    if len(images) == 0:
        print("Nu am găsit nicio imagine.")
        return

    train_val, test = train_test_split(images, test_size=0.2, stratify=[x["dx"] for x in images], random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, stratify=[x["dx"] for x in train_val], random_state=42)

    splits = {"train": train, "val": val, "test": test}

    for split_name, split_images in splits.items():
        for cls in CLASSES:
            out_cls_dir = os.path.join(OUTPUT_DIR, split_name, cls)
            os.makedirs(out_cls_dir, exist_ok=True)

        for img in split_images:
            dst = os.path.join(OUTPUT_DIR, split_name, img["dx"], os.path.basename(img["image_path"]))
            shutil.copy2(img["image_path"], dst)

    print("\n Imaginile au fost împărțite si copiate în folderul splits:")
    for split_name in splits:
        print(f"  - {split_name}: {len(splits[split_name])} imagini")
if __name__ == "__main__":
    main()