# Importing libraries
import glob
import os

import lancedb
import numpy as np
import open_clip
import pandas as pd
import pyarrow as pa
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor


# Dataset and dataloader
class KeyframesDataset(Dataset):
    def __init__(self, preprocess):
        self.PATH = "keyframes"
        print(f"Image path for dataset: {self.PATH}\\*\\*")
        self.images_paths = glob.glob(f"{self.PATH}\\*\\*")
        self.preprocess = preprocess

        map_keyframes_paths = glob.glob('.\\map-keyframes\\*.csv')
        self.map_keyframes_dfs = {}
        for path in map_keyframes_paths:
            self.map_keyframes_dfs[path.rsplit('\\', 1)[1][:-4]] = pd.read_csv(path)
        
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.images_paths[idx]))
        video_name = self.images_paths[idx].rsplit('\\', 2)[1]
        image_name = self.images_paths[idx].rsplit('\\', 1)[1]
        frame_idx = self.map_keyframes_dfs[video_name].at[int(image_name[:-4])-1, 'frame_idx']
        return image, video_name, image_name, frame_idx, self.images_paths[idx]

if __name__ == '__main__':
    # Load CLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    model.to(device)

    dataset = KeyframesDataset(preprocess)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=12, persistent_workers=True)
    print("Number of images: ", len(dataset))
    print("Number of batches: ", len(dataloader))


    # Create lancedb instance
    lancedb_instance = lancedb.connect("database.lance")
    TABLE_NAME = "patch14v2_openclip"
    if TABLE_NAME in lancedb_instance.table_names():
        database = lancedb_instance[TABLE_NAME]
        print(f"Warning: Table {TABLE_NAME} already exists!")
    else:
        schema = pa.schema([
            pa.field("embedding", pa.list_(pa.float32(), 1024)),
            pa.field("video_name", pa.string()),
            pa.field("image_name", pa.string()),
            pa.field("frame_idx", pa.int32()),
            pa.field("path", pa.string()),
        ])
        lancedb_instance.create_table(TABLE_NAME, schema=schema)
        database = lancedb_instance[TABLE_NAME]

    LEN_DATALOADER = len(dataloader)
    SAVE_EVERY = int(0.05 * LEN_DATALOADER)

    df = pd.DataFrame(columns=['embedding', 'video_name', 'image_name', 'frame_idx', 'path'])
    for i, (images, video_names, image_names, frame_idxs, paths) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            embeddings = model.encode_image(images)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.squeeze().cpu().numpy()
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