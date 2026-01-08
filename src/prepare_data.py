import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# HAM10000 pe D:
IMAGES_DIR_1 = r"D:\HAM10000\HAM10000_images_part_1"
IMAGES_DIR_2 = r"D:\HAM10000\HAM10000_images_part_2"
CSV_PATH = r"D:\HAM10000\HAM10000_metadata.csv"

# OUTPUT rămâne în proiectul PyCharm
OUTPUT_DIR = "../data/splits"

# 4 clase selectate
CLASSES = ["mel", "nv", "bcc", "bkl"]

def find_image(image_id):
    for folder in [IMAGES_DIR_1, IMAGES_DIR_2]:
        path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None

def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["dx"].isin(CLASSES)]

    images = []
    for _, row in df.iterrows():
        img_path = find_image(row["image_id"])
        if img_path:
            images.append({
                "image_path": img_path,
                "dx": row["dx"]
            })

    print("Total imagini:", len(images))

    train_val, test = train_test_split(
        images,
        test_size=0.2,
        stratify=[x["dx"] for x in images],
        random_state=42
    )

    train, val = train_test_split(
        train_val,
        test_size=0.1,
        stratify=[x["dx"] for x in train_val],
        random_state=42
    )

    splits = {"train": train, "val": val, "test": test}

    for split_name, split_images in splits.items():
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split_name, cls), exist_ok=True)

        for img in split_images:
            dst = os.path.join(
                OUTPUT_DIR,
                split_name,
                img["dx"],
                os.path.basename(img["image_path"])
            )
            shutil.copy2(img["image_path"], dst)

    for split in splits:
        print(f"{split}: {len(splits[split])} imagini")

if __name__ == "__main__":
    main()
