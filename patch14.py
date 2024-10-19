import lancedb
import pyarrow as pa
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


class find_query_text:
    def __init__(self, link_database):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.text_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True).to(self.device)
        self.lancedb_instance = lancedb.connect(fr"{link_database}")
        if "patch14" in self.lancedb_instance.table_names():
            self.database = self.lancedb_instance["patch14"]
        else:
            schema = pa.schema([
                pa.field("embedding", pa.list_(pa.float32(), 768)),
                pa.field("caption_embedding", pa.list_(pa.float32(), 768)),
                pa.field("path", pa.string()),
                pa.field("caption", pa.string())
            ])
            self.lancedb_instance.create_table("patch14", schema=schema)
            self.database = self.lancedb_instance["patch14"]
        if "video" not in self.lancedb_instance.table_names():
            schema = pa.schema([
                pa.field("summary_embedding", pa.list_(pa.float32(), 768)),
                pa.field("name",pa.string()),
                pa.field("summary", pa.string()),
            ])
            self.lancedb_instance.create_table("video", schema=schema)

    def add_to_database_video(self, summaries_list):
        table = self.lancedb_instance['video']
        data = []
        for name, summary in summaries_list:
            summary_embedding = self.text_model.encode(summary).flatten()
            data.append({
                "summary_embedding": summary_embedding,
                "name": name,
                "summary": summary
            })
        table.add(data)

    def find_video_by_summary(self, query: str, num_data=25, metric = "cosine"):
        table = self.lancedb_instance['video']
        
        summary_embedding = self.text_model.encode(query).flatten()
        
        results = table.search(summary_embedding, vector_column_name='summary_embedding').metric(metric).limit(num_data).to_list()

        pathlist = []
        for result in results:
            pathlist.append(result['name'])
            print(result['name'])
        return pathlist

    def add_to_database(self, link_pictures = [], captions = []):
        data = []
        
        if len(captions) != len(link_pictures):
            print("data mismatch")
            return

        for picture, caption in zip(link_pictures, captions):
            path = fr"{picture}"

            table = self.database
            existing = table.to_pandas().query('path == @path')
            if not existing.empty:
                print(f"Path already exists in database, skipping: {path}")
                continue

            image = Image.open(path)

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs).cpu().squeeze().numpy()


            caption_embedding = self.text_model.encode(caption).flatten()

            data.append({"embedding": embedding,"caption_embedding": caption_embedding, "path": path, "caption": caption})
            print(f"save embedding for: {path}")

        if data:
            self.database.add(data)
        else:
            print("No data to add.")


    def find_query_by_embedding(self, query, num_data=25, metric="cosine"):
        if self.database is None:
            print("Database is empty!")
            return []
        search_text = [query]
        inputs = self.processor(text=search_text, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs).cpu().squeeze().numpy()

        results = self.database.search(text_embeddings, vector_column_name='embedding').metric(metric).limit(num_data).to_list()

        pathlist = []
        for result in results:
            pathlist.append(result['path'])
            print(result['path'])
        
        return pathlist


    def find_query_by_picture(self, link_picture, num_data=25, metric="cosine"):
        if self.database is None:
            print("Database is empty!")
            return []


        image = Image.open(link_picture)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs).cpu().squeeze().numpy()

        results = self.database.search(embedding, vector_column_name='embedding').metric(metric).limit(num_data).to_list()

        pathlist = []
        for result in results:
            pathlist.append(result['path'])
            print(result['path'])
        
        return pathlist


    def find_query_by_caption(self, query, num_data=25, metric = "cosine"):
        if self.database is None:
            print("Database is empty!")
            return []
        
        caption_embedding = self.text_model.encode(query).flatten()
        
        results = self.database.search(caption_embedding, vector_column_name='caption_embedding').metric(metric).limit(num_data).to_list()

        pathlist = []
        for result in results:
            pathlist.append(result['path'])
            print(result['path'])
        return pathlist


