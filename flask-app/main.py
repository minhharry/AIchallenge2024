import io
import json
import os
from typing import Union

import lancedb
import open_clip
import pandas as pd
import requests
import torch
from deep_translator import GoogleTranslator
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, session, url_for)
from PIL import Image

from flask_session import Session

app = Flask(__name__)
app.secret_key = 'Aoi_sutato'
PATH = "E:\\Downloads\\atm24949_withsound\\keyframes"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

class LancedbModel():
    def __init__(self) -> None:
        self.lancedb_instance = lancedb.connect("database.lance")
        self.database = {}
        if "patch14v2_all" in self.lancedb_instance.table_names():
            self.database['openclip'] = self.lancedb_instance["patch14v2_all"]
            print(f"Load database success!")
        else:
            print(f"Load database error!")
            raise FileExistsError("Database not found!")
        
        # if "patch14v2_extended" in self.lancedb_instance.table_names():
        #     self.database['clip'] = self.lancedb_instance["patch14v2_extended"]
        # else:
        #     print(f"Load database error!")
        #     raise FileExistsError("Database not found!")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_openclip, _, self.preprocess_openclip = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
        self.model_openclip.eval()  
        self.model_openclip.to(self.device)
        self.tokenizer_openclip = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
        print(f"Load openclip model success!")
        # self.model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        # self.processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def _inference_text(self, query, model_name):
        if model_name not in ["clip", "openclip"]:
            raise ValueError("Invalid model name. Must be 'clip' or 'openclip'.")
        
        if model_name == "openclip":
            inputs = self.tokenizer_openclip([query]).to(self.device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                text_embedding = self.model_openclip.encode_text(inputs)
                text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
                text_embedding = text_embedding.squeeze().cpu().numpy()
            return text_embedding
        
        if model_name == "clip":
            inputs = self.processor_clip(text=query, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                text_embedding = self.model_clip.get_text_features(**inputs).squeeze().cpu().numpy()
            return text_embedding
        
        return None
    
    def _inference_image(self, query: Image.Image, model_name):
        if model_name not in ["clip", "openclip"]:
            raise ValueError("Invalid model name. Must be 'clip' or 'openclip'.")

        if model_name == "openclip":
            query = self.preprocess_openclip(query).unsqueeze(0)
            query = query.to(self.device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                embedding = self.model_openclip.encode_image(query)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                embedding = embedding.squeeze().cpu().numpy()
            return embedding
        
        if model_name == "clip":
            inputs = self.processor_clip(images=query, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self.model_clip.get_image_features(**inputs).cpu().squeeze().numpy()
            return embedding
        
        return None

    def find_by_text(self, query: str, num_data=25, metric="cosine", sql_filter : None | str = None, model_name: str = 'openclip') -> pd.DataFrame:
        text_embedding = self._inference_text(query, model_name)
        if sql_filter == None:
            results = self.database[model_name].search(text_embedding, vector_column_name='embedding').metric(metric).limit(num_data).to_pandas()
        else:
            results = self.database[model_name].search(text_embedding, vector_column_name='embedding').where(sql_filter, prefilter=True).metric(metric).limit(num_data).to_pandas()
        return results

    
    def find_by_picture(self, query: Union[Image.Image, str], num_data=25, metric="cosine", sql_filter: None | str = None, model_name: str = 'openclip') -> pd.DataFrame:
        if isinstance(query, str):
            query = Image.open(query)
        elif not isinstance(query, Image.Image):
            raise TypeError("query must be a PIL Image object or a string representing an image path")

        embedding = self._inference_image(query, model_name)
        if sql_filter == None:        
            results = self.database[model_name].search(embedding, vector_column_name='embedding').metric(metric).limit(num_data).to_pandas()
        else:
            results = self.database[model_name].search(embedding, vector_column_name='embedding').where(sql_filter, prefilter=True).metric(metric).limit(num_data).to_pandas()
        return results

translator = GoogleTranslator(source='vi', target='en')
model = LancedbModel()
with open('fps30.json', 'r') as f:
    fps30 = json.load(f)
media_info_df = pd.read_csv('media-info.csv')

def find_by_2_query(query1: str, query2: str, num_data=5, model_name='openclip', sql_filter: None | str = None):
    global model
    df1 = model.find_by_text(query1, num_data=num_data, model_name=model_name, sql_filter=sql_filter)
    df2 = pd.DataFrame()
    for video_name in df1['video_name'].unique():
        temp_df = model.find_by_text(query2, num_data=num_data, sql_filter=f"video_name = '{video_name}'")
        df2 = pd.concat([df2, temp_df], ignore_index=True)
    return df1, df2

def get_ytb_link(video_name:str, keyframe_number: int):
    global media_info_df
    timestamp = keyframe_number // 25
    if video_name in fps30:
        timestamp = keyframe_number // 30
    if video_name in ["L24_V044"]:
        timestamp = int(keyframe_number / 26.44)
    timestamp = max(0, timestamp - 2)
    for index, row in media_info_df.iterrows():
        if row['filename'] == video_name + '.json':
            return row['watch_url']+f'&t={timestamp}s'
    return "Video not found!"

@app.route('/')
def index():
    if 'DATA' not in session:
        DATA = [
            [
                "Query 1: , Query 2: \n",
                {'L01_V001': [('image-01-scene-001-frame-1.jpg', 1, get_ytb_link('L01_V001', 1), 'text-blue-700')]}
            ]
        ]
        session["num_data"] = 100
        return render_template('index.html', data=DATA, num_data=100)
        
    return render_template('index.html', data=session["DATA"], num_data=session["num_data"])

@app.route('/session_clear', methods=['POST'])
def session_clear():
    session.clear()
    return redirect(url_for('index'))

def process_text_query(first_query: str, second_query: str, sql_filter: None | str = None, num_data: int = 200):
    global model
    DATA = {}
    if first_query != "" and second_query == "":
        df = model.find_by_text(first_query, num_data=num_data, sql_filter=sql_filter)
        DATA = {}
        for index, row in df.iterrows():
            if row['video_name'] not in DATA:
                DATA[row['video_name']] = []
            DATA[row['video_name']].append([row['image_name'], row['frame_idx'], get_ytb_link(row['video_name'], row['frame_idx']), 'text-blue-500'])
    if first_query != "" and second_query != "":
        df1, df2 = find_by_2_query(first_query, second_query, num_data=num_data, sql_filter=sql_filter)
        df1['df1'] = 'df1'
        df = pd.concat([df1, df2], ignore_index=True)
        DATA = {}
        for index, row in df.iterrows():
            if row['video_name'] not in DATA:
                DATA[row['video_name']] = []
            DATA[row['video_name']].append([row['image_name'], row['frame_idx'], get_ytb_link(row['video_name'], row['frame_idx']),'text-blue-500' if row['df1'] == 'df1' else 'text-orange-500'])
    if first_query == "" and second_query == "" and sql_filter != "":
        df = model.database['openclip'].search().where(sql_filter).limit(num_data).to_pandas()
        DATA = {}
        for index, row in df.iterrows():
            if row['video_name'] not in DATA:
                DATA[row['video_name']] = []
            DATA[row['video_name']].append([row['image_name'], row['frame_idx'], get_ytb_link(row['video_name'], row['frame_idx']), 'text-blue-500'])
    return DATA

@app.route('/text-query', methods=['POST'])
def text_query():
    if 'DATA' not in session:
        session["DATA"] = []
    first_query = request.form.get('firstQuery')
    second_query = request.form.get('secondQuery')
    sql_filter = request.form.get('sqlFilter')
    num_data = int(request.form.get('numData'))
    session["num_data"] = num_data
    session["DATA"].append([
        f"Query 1: {first_query}, Query 2: {second_query}, SQL Filter: {sql_filter} ----------"
    ])
    if first_query != "":
        first_query = translator.translate(first_query)
    if second_query != "":
        second_query = translator.translate(second_query)
    if sql_filter == "":
        sql_filter = None
    session["DATA"][-1][0] += f"Query 1: {first_query}, Query 2: {second_query}, SQL Filter: {sql_filter}\n"
    DATA = process_text_query(first_query, second_query, sql_filter, num_data)
    session["DATA"][-1].append(DATA)
    return redirect(url_for('index'))

@app.route('/file-query', methods=['POST'])
def file_query():
    if 'DATA' not in session:
        session["DATA"] = []
    uploaded_files = request.files.getlist("files")
    num_data = session["num_data"]
    for file in uploaded_files:
        if file and file.filename.lower().endswith('.txt'):
            content = file.read().decode('utf-8')
            content_translated = translator.translate(content)
            DATA = process_text_query(content_translated, "", None, num_data)
            session["DATA"].append([
                f'Query text file: {file.filename} | {content} | {content_translated}',
                DATA
            ])
        elif file and file.filename.lower().endswith(('blob', '.jpeg', '.jpg', '.png')):
            session["DATA"].append([
                f'Query Image: {file.filename}\n'
            ])
            image_data = file.read()
            image = Image.open(io.BytesIO(image_data))
            df = model.find_by_picture(image, num_data=num_data)
            DATA = {}
            for index, row in df.iterrows():
                if row['video_name'] not in DATA:
                    DATA[row['video_name']] = []
                DATA[row['video_name']].append([row['image_name'], row['frame_idx'], get_ytb_link(row['video_name'], row['frame_idx']), 'text-blue-500'])
            session["DATA"][-1].append(DATA)
        else:
            session["DATA"].append([
                f'Query file: {file.filename} | {file.filename} is not in (.txt, .jpeg, .jpg, .png)',
                {}
            ])
    return redirect(url_for('index'))

@app.route('/images/<path:path>/<image_name>')
def get_image(path, image_name):
    # image_url = f'https://aichallenge.uwuganyudesu.online/images/{path}/{image_name}'  # Adjust URL as needed
    # response = requests.get(image_url)
    # if response.status_code == 200:
    #     return response.content
    # else:
    #     return "Error fetching image"
    global PATH
    return send_from_directory(os.path.join(PATH, path), image_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
