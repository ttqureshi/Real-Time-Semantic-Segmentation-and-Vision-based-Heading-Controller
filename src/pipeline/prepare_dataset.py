import os
import shutil
import zipfile
from pathlib import Path
import gdown

def download_and_extract():
    file_id = '1nNtlvAEsqXF0UB1cyETenuka3-tSeyah'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'data/dataset.zip'
    
    os.makedirs('data/raw', exist_ok=True)
    
    if not os.path.exists(output):
        print("Downloading dataset from Google Drive...")
        gdown.download(url, output, quiet=False)
    else:
        print("Dataset zip already exists.")
        
    print("Extracting dataset...")
    # Clear out previous extraction if it exists to avoid conflicts
    raw_dest = Path('data/raw/roboflow_dataset')
    if raw_dest.exists():
        shutil.rmtree(raw_dest)
        
    os.makedirs(raw_dest, exist_ok=True)
        
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(raw_dest)
        
def organize_dataset():
    raw_dir = Path("data/raw/roboflow_dataset")
    
    # Find the 'train' folder to determine the base directory of the dataset inside the zip
    train_dirs = list(raw_dir.rglob("train"))
    if not train_dirs:
        print("Could not find 'train' directory in extracted dataset.")
        return
        
    base_dataset_dir = train_dirs[0].parent
    processed_dir = Path("data/processed")
    
    splits_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }
    
    for raw_split, proc_split in splits_mapping.items():
        split_dir = base_dataset_dir / raw_split
        if not split_dir.exists():
            print(f"Warning: Raw split directory {raw_split} not found. Skipping.")
            continue
            
        images_out_dir = processed_dir / proc_split / "images"
        masks_out_dir = processed_dir / proc_split / "masks"
        
        os.makedirs(images_out_dir, exist_ok=True)
        os.makedirs(masks_out_dir, exist_ok=True)
        
        images = list(split_dir.glob("*.jpg"))
        copied_count = 0
        
        for img_path in images:
            mask_name = img_path.name.replace(".jpg", "_mask.png")
            mask_path = split_dir / mask_name
            
            if not mask_path.exists():
                print(f"Warning: Mask not found for {img_path.name}")
                continue
                
            shutil.copy2(img_path, images_out_dir / img_path.name)
            shutil.copy2(mask_path, masks_out_dir / mask_path.name)
            copied_count += 1
            
        print(f"Copied {copied_count} files for {proc_split} split.")

if __name__ == "__main__":
    download_and_extract()
    organize_dataset()
    print("Dataset preparation complete!")
