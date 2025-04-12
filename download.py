import os
import requests
from tqdm import tqdm

base_url = "https://hf-mirror.com/datasets/HuggingFaceM4/COCO/resolve/refs%2Fconvert%2Fparquet/2014_captions"
output_dir = "./.cache/huggingface/datasets/HuggingFaceM4___coco/validation_files"

os.makedirs(output_dir, exist_ok=True)

for i in tqdm(range(2), desc="Downloading validation set"):
    file_name = f"coco-validation-{str(i).zfill(5)}-of-00002.parquet"
    url = f"{base_url}/{file_name}"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(os.path.join(output_dir, file_name), "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

print("Download complete!")
