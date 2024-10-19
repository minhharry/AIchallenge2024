import glob
from PIL import Image

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

prompt = "<MORE_DETAILED_CAPTION>"

def image_captioning(image_path):
  data = []
  image = Image.open(image_path)
  inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
  generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3,
      do_sample=False
  )
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

  parsed_answer = processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>", image_size=(image.width, image.height))
  data = {'frame' : image_path.rsplit("\\",1)[1] , 'caption' : parsed_answer["<MORE_DETAILED_CAPTION>"]}
  return data
import csv
def write_to_csv(data, file_path, fieldnames=None):
  """Writes data to a CSV file.

  Args:
      data (list[dict] or list[list]): The data to write, as a list of dictionaries or lists.
      file_path (str): The path to the CSV file.
      fieldnames (list[str], optional): The field names for the CSV file. If not provided,
          the first row of data will be used as fieldnames.
  """

  with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)


from tqdm import tqdm
import os
i=1
num = str(i).zfill(2)
base = f"E:\\Downloads\\atm24949\\keyframes\\L{num}*"
dirs = glob.glob(base)
os.makedirs(f"./L{num}", exist_ok=True)
for dir in tqdm(dirs):
    v_name = dir.rsplit("\\",1)[1]
    if os.path.exists(f'./L{num}/'+v_name+'.csv'):
        print("SKIP ", f'./L{num}/'+v_name+'.csv')
        continue
    data = []
    files = glob.glob(dir+'\\*.jpg')
    for file in files:
        data.append(image_captioning(file))
    
    write_to_csv(data, f'./L{num}/'+v_name+'.csv', fieldnames=['frame', 'caption'])


i=7
num = str(i).zfill(2)
base = f"E:\\Downloads\\atm24949\\keyframes\\L{num}*"
dirs = glob.glob(base)
os.makedirs(f"./L{num}", exist_ok=True)
for dir in tqdm(dirs):
    v_name = dir.rsplit("\\",1)[1]
    if os.path.exists(f'./L{num}/'+v_name+'.csv'):
        print("SKIP ", f'./L{num}/'+v_name+'.csv')
        continue
    data = []
    files = glob.glob(dir+'\\*.jpg')
    for file in files:
        data.append(image_captioning(file))
    
    write_to_csv(data, f'./L{num}/'+v_name+'.csv', fieldnames=['frame', 'caption'])