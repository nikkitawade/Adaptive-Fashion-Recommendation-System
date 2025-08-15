import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import open_clip
from collections import defaultdict
from sklearn.model_selection import train_test_split

#Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
clip_model = clip_model.to(device)
clip_model.eval()

#Function to extract image embeddings using CLIP
def extract_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
        return image_features.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Image error ({image_path}): {e}")
        return None

#Process a single JSON file and extract attributes + image embedding
def process_json_file(file_path, images_folder):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        main_data = data.get('data', {})
        if not main_data:
            return None

        #Basic product info
        product_id = main_data.get('id') or os.path.splitext(os.path.basename(file_path))[0]

        attributes = {
            'product_id': product_id,
            'product_name': main_data.get('productDisplayName'),
            'brand': main_data.get('brandName'),
            'category': main_data.get('masterCategory', {}).get('typeName'),
            'sub_category': main_data.get('subCategory', {}).get('typeName'),
            'article_type': main_data.get('articleType', {}).get('typeName'),
            'color': main_data.get('baseColour', '').lower(),
            'season': main_data.get('season', '').lower(),
            'year': main_data.get('year'),
            'usage': main_data.get('usage', '').lower(),
            'pattern': main_data.get('articleAttributes', {}).get('Pattern', '').lower(),
            'sleeve_length': main_data.get('articleAttributes', {}).get('Sleeve Length', '').lower(),
            'dress_length': main_data.get('articleAttributes', {}).get('Length', '').lower(),
            'fabric': main_data.get('articleAttributes', {}).get('Fabric', '').lower(),
            'neck_type': main_data.get('articleAttributes', {}).get('Neck', '').lower(),
            'price': main_data.get('price'),
            'discounted_price': main_data.get('discountedPrice'),
            'gender': main_data.get('gender', '').lower(),
            'age_group': main_data.get('ageGroup', '').lower(),
            'image_url': main_data.get('styleImages', {}).get('default', {}).get('imageURL'),
            'usage_group': main_data.get('usage', '').lower(),
            'local_image_path': None,
            'image_embedding': None
        }

        #Find local image and extract embedding
        for ext in ('.jpg', '.png', '.jpeg'):
            possible_path = os.path.join(images_folder, f"{product_id}{ext}")
            if os.path.exists(possible_path):
                attributes['local_image_path'] = possible_path
                attributes['image_embedding'] = extract_image_embedding(possible_path)
                break

        #Store available sizes
        size_options = [f"{opt.get('name', '')}:{opt.get('value', '')}"
                        for opt in main_data.get('styleOptions', [])]
        attributes['sizes'] = '|'.join(size_options)

        #Add other flags from metadata
        for flag in main_data.get('otherFlags', []):
            attributes[flag.get('name', 'unknown_flag').lower()] = flag.get('value', '')

        return attributes

    except Exception as e:
        print(f"JSON error ({file_path}): {e}")
        return None

#Split JSON files into train/test by category & article type
def split_json_by_category(metadata_folder, test_size=0.2, min_group_size=5):
    print(f"\nScanning metadata in: {metadata_folder}")
    all_files = [f for f in os.listdir(metadata_folder) if f.endswith(".json")]
    print(f"Total JSON files found: {len(all_files)}")

    category_map = defaultdict(list)
    with tqdm(total=len(all_files), desc="Processing products") as pbar:
        for f in all_files:
            path = os.path.join(metadata_folder, f)
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    data = json.load(file).get("data", {})
                    category = data.get('masterCategory', {}).get('typeName', 'Unknown')
                    article = data.get('articleType', {}).get('typeName', 'Unknown')
                    key = (category, article)
                    category_map[key].append(f)
                    pbar.update(1)
            except Exception as e:
                print(f"Error parsing {f}: {e}")

    print(f"\nGrouped into {len(category_map)} category-article_type combinations")

    train_files, test_files = [], []

    #Create splits for each category group
    for key, files in category_map.items():
        category, article = key
        print(f"Processing group: {category} / {article} | Items: {len(files)}")

        if len(files) < min_group_size:
            print("   🔸 Too small, all going to train set.")
            train_files.extend(files)
        else:
            train, test = train_test_split(files, test_size=test_size, random_state=42)
            print(f"Train: {len(train)} | Test: {len(test)}")
            train_files.extend(train)
            test_files.extend(test)

    print(f"\nFinal split => Train: {len(train_files)} files | Test: {len(test_files)} files")
    return train_files, test_files

#Process files and save embeddings to pickle
def process_and_save_split(file_list, split_name, images_folder, metadata_folder, output_path):
    print(f"\nStarting processing for {split_name} split...")
    processed_data = []
    for fname in tqdm(file_list, desc=f"Extracting {split_name}"):
        full_path = os.path.join(metadata_folder, fname)
        data = process_json_file(full_path, images_folder)
        if data and data['image_embedding'] is not None:
            processed_data.append(data)

    #Save to pickle file
    if processed_data:
        df = pd.DataFrame(processed_data)
        df = df.drop_duplicates('product_id')
        df.to_pickle(output_path)
        print(f"Saved {split_name} split with {len(df)} items at {output_path}")
    else:
        print(f"No items extracted for {split_name} split.")

#Main function to prepare dataset
def process_dataset():
    dataset_path = "fashion-dataset"
    images_folder = os.path.join(dataset_path, "images")
    metadata_folder = os.path.join(dataset_path, "styles")

    os.makedirs("output", exist_ok=True)

    print("Starting dataset preparation...\n")
    train_files, test_files = split_json_by_category(metadata_folder, test_size=0.2)

    process_and_save_split(
        train_files, "train", images_folder, metadata_folder, "output/fashion_train_embeddings.pkl"
    )
    process_and_save_split(
        test_files, "test", images_folder, metadata_folder, "output/fashion_test_embeddings.pkl"
    )

    print("\nDataset is ready.\n")

#Run the dataset processing
if __name__ == "__main__":
    process_dataset()
