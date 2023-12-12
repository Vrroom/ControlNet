import sys
from tqdm import tqdm
import json
import os
import random
sys.path.insert(0, '/home/ubuntu/miniconda3/envs/control/lib/python3.8/site-packages/')
import torch
from PIL import Image
import argparse
from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device=device
)

def aspectRatioPreservingResize (pil_img, smaller_dim) :
    """ utility for resizing image, ensuring that smaller dimension matches """
    h, w = pil_img.size
    if h < w :
        h, w = smaller_dim, smaller_dim * w / h
    else :
        h, w = smaller_dim * h / w, smaller_dim
    h, w = int(h), int(w)
    resized = pil_img.resize((h, w))
    return resized

def caption_image(img_path, n_captions) : 
    try : 
        img = aspectRatioPreservingResize(Image.open(img_path).convert("RGB") , 512)
        image = vis_processors["eval"](img).unsqueeze(0).to(device)
        captions = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=n_captions)
        return captions
    except Exception :
        return []

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='Caption a subset of files')
    parser.add_argument('--image_dir', type=str, help='Base dir containing images')
    parser.add_argument('--file_list', type=str, help='Text file containing list of files')
    parser.add_argument('--n_samples', type=int, help='Number of samples')
    parser.add_argument('--n_captions', type=int, help='How many captions to generate')
    parser.add_argument('--out_json', type=str, help='Where to dump the captions')

    # Parse the arguments
    args = parser.parse_args()
    with open(args.file_list) as fp :
        img_paths = fp.readlines()
    img_paths = [_.strip() for _ in img_paths]
    img_paths = random.sample(img_paths, k=args.n_samples)
    img_paths = [os.path.abspath(os.path.join(args.image_dir, _)) for _ in img_paths]
    captions = [caption_image(_, args.n_captions) for _ in tqdm(img_paths)] 
    with open(args.out_json, 'w+') as fp : 
        json.dump(dict(zip(img_paths, captions)), fp)

