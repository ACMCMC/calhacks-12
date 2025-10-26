from huggingface_hub import login
from datasets import load_dataset
import os
from PIL import Image

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)

dataset = load_dataset("PeterBrendan/AdImageNet")

save_dir = "./data/images"
os.makedirs(save_dir, exist_ok=True)

for i, example in enumerate(dataset["train"]):
    img = example["image"]
    fmt = img.format.lower() if img.format else "png"

    # build filename
    filename = f"image_{i}.{fmt}"
    filepath = os.path.join(save_dir, filename)

    # save the image
    img.save(filepath)
    
    