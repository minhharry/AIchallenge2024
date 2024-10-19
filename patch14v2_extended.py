# Importing libraries
import glob
import os

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor


# Dataset and dataloader
class KeyframesDataset(Dataset):
    def __init__(self):
        self.PATH = "E:\\Downloads\\atm24949_withsound\\keyframes"
        print(f"Image path for dataset: {self.PATH}\\*\\*")
        self.images_paths = glob.glob(f"{self.PATH}\\*\\*")
        
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = read_image(self.images_paths[idx])
        video_name = self.images_paths[idx].rsplit('\\', 2)[1]
        image_name = self.images_paths[idx].rsplit('\\', 1)[1]
        frame_idx = self.images_paths[idx].rsplit('\\', 1)[1][:-4].rsplit('-', 1)[1]
        return image, video_name, image_name, frame_idx, self.images_paths[idx].split('\\', 3)[3]

if __name__ == "__main__":
    # Create lancedb instance
    lancedb_instance = lancedb.connect("database.lance")
    TABLE_NAME = "patch14v2_extended"
    if TABLE_NAME in lancedb_instance.table_names():
        database = lancedb_instance[TABLE_NAME]
        print(f"Warning: Table {TABLE_NAME} already exists!")
    else:
        schema = pa.schema([
            pa.field("embedding", pa.list_(pa.float32(), 768)),
            pa.field("video_name", pa.string()),
            pa.field("image_name", pa.string()),
            pa.field("frame_idx", pa.int32()),
            pa.field("path", pa.string()),
        ])
        lancedb_instance.create_table(TABLE_NAME, schema=schema)
        database = lancedb_instance[TABLE_NAME]
        
    dataset = KeyframesDataset()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=6)
    print("Number of images: ", len(dataset))
    print("Number of batches: ", len(dataloader))

    # Load CLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    LEN_DATALOADER = len(dataloader)
    SAVE_EVERY = int(0.05 * LEN_DATALOADER)

    df = pd.DataFrame(columns=['embedding', 'video_name', 'image_name', 'frame_idx', 'path'])

    for i, (images, video_names, image_names, frame_idxs, paths) in enumerate(tqdm(dataloader)):
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            embeddings = model.get_image_features(**inputs).cpu().squeeze().numpy()
        data = {
            'embedding': [],
            'video_name': [],
            'image_name': [],
            'frame_idx': [],
            'path': []
        }
        for embedding, video_name, image_name, frame_idx, path in zip(embeddings, video_names, image_names, frame_idxs, paths):
            data['embedding'].append(embedding)
            data['video_name'].append(video_name)
            data['image_name'].append(image_name)
            data['frame_idx'].append(int(frame_idx))
            data['path'].append(path)
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        if (i + 1) % SAVE_EVERY == 0 or i + 1 == LEN_DATALOADER:
            lancedb_instance[TABLE_NAME].add(df)
            df = pd.DataFrame(columns=['embedding', 'video_name', 'image_name', 'frame_idx', 'path'])
            print(f"Saved embeddings for batch {i+1}/{LEN_DATALOADER}")