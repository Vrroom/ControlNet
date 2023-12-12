import sys
from tqdm import tqdm
from more_itertools import chunked
import os
import argparse
import numpy as np
sys.path.insert(0, '/home/ubuntu/miniconda3/envs/control/lib/python3.8/site-packages/')
import torch
import clip
from PIL import Image
import os.path as osp
import ray

ray.init()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

def splitter(pairs) :
    a_s = [a for a, b in pairs]
    b_s = [b for a, b in pairs]
    return a_s, b_s

@torch.no_grad() 
def get_embedding (preprocessed_images) : 
    image = torch.stack(preprocessed_images).to(device)
    image_features = model.encode_image(image)
    return image_features.squeeze().detach().cpu().numpy()

@ray.remote
def preprocess_images (path) :
    try: 
        return preprocess(Image.open(path)), path
    except Exception :
        return None, path

def compute_and_save (args, processed_images, paths, chunk_id) :
    xs = get_embedding(processed_images)
    with open(osp.join(args.dump_dir, args.out_txt_pref + '_' + str(chunk_id).zfill(7) + '.txt'), 'w+') as fp :
        fp.write('\n'.join(paths))
    xs.dump(osp.join(args.dump_dir, args.out_npy_pref + '_' + str(chunk_id).zfill(7) + '.npy'))

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='Create a CLIP index for images')
    parser.add_argument('--image_dir', type=str, help='Base dir containing images')
    parser.add_argument('--file_list', type=str, help='Text file containing list of files')
    parser.add_argument('--dump_dir', type=str, help='Where to jump chunks')
    parser.add_argument('--out_npy_pref', type=str, help='Where to dump the index')
    parser.add_argument('--out_txt_pref', type=str, help='Where to dump the successful image paths')
    # Parse the arguments
    args = parser.parse_args()
    with open(args.file_list) as fp :
        img_paths = fp.readlines()
    img_paths = [osp.abspath(osp.join(args.image_dir, _.strip())) for _ in img_paths]
    chunked_img_paths = list(chunked(img_paths, 400))
    for chunk_id, chunk in enumerate(tqdm(chunked_img_paths)):
        futures = [preprocess_images.remote(_) for _ in chunk]
        results = ray.get(futures)
        processed_images = [a for a, b in results if a is not None]
        paths = [b for a, b in results if a is not None] 
        compute_and_save(args, processed_images, paths, chunk_id)
