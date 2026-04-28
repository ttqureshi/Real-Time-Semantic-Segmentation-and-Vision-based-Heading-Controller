import os
import shutil
from pathlib import Path

def prepare_dataset():
    raw_dir = Path("data/raw/roboflow_dataset")
    processed_dir = Path("data/processed")
    
    # Mapping Roboflow splits to our project splits
    splits_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }
    
    for raw_split, proc_split in splits_mapping.items():
        split_dir = raw_dir / raw_split
        if not split_dir.exists():
            print(f"Warning: Raw split directory {raw_split} not found. Skipping.")
            continue
            
        images_out_dir = processed_dir / proc_split / "images"
        masks_out_dir = processed_dir / proc_split / "masks"
        
        os.makedirs(images_out_dir, exist_ok=True)
        os.makedirs(masks_out_dir, exist_ok=True)
        
        images = list(split_dir.glob("*.jpg"))
        
        for img_path in images:
            mask_name = img_path.name.replace(".jpg", "_mask.png")
            mask_path = split_dir / mask_name
            
            if not mask_path.exists():
                print(f"Warning: Mask not found for {img_path.name}")
                continue
                
            shutil.copy2(img_path, images_out_dir / img_path.name)
            shutil.copy2(mask_path, masks_out_dir / mask_path.name)
            
        print(f"Copied {len(images)} files for {proc_split} split.")

if __name__ == "__main__":
    prepare_dataset()
