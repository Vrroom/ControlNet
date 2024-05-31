import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import os
import os.path as osp
from glob import glob
import re
import random
import pytorch_lightning as pl
import json
import cv2
import skimage
import warnings
from ldm.util import log_txt_as_img
from imageOps import *

warnings.filterwarnings('error', message='Corrupt EXIF data.*')

MIDJOURNEY_MSGS_PARQUET_FILE = '/home/ubuntu/general-purpose/midjourney-messages/data'
CONCEPTUAL_CAPTIONS_CSV_FILE = '/home/ubuntu/general-purpose/CC/ConceptualCaptions.csv'

def no_op (*args, **kwargs) : 
    pass

def clamp (x, l, h) : 
    return min(max(x, l), h)

def txt_as_pil(wh, x, size=10):
    txt = Image.new("RGB", wh, color="white")
    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
    nc = int(40 * (wh[0] / 256))
    lines = "\n".join(x[start:start + nc] for start in range(0, len(x), nc))
    try:
        draw.text((0, 0), lines, fill="black", font=font)
    except UnicodeEncodeError:
        print("Cant encode string for logging. Skipping.")
    return txt

def hint_to_pil_images(hints):
    hints = torch.clamp((hints * 255), 0.0, 255.0).to(torch.uint8)
    hints = hints.detach().cpu().numpy() 

    B, *_ = hints.shape

    edge_map_np = hints[..., 0]
    edge_map_images = [Image.fromarray(edge_map_np[i], mode='L') for i in range(B)]

    color_hint_np = hints[..., 1:]
    color_hint_images = [Image.fromarray(color_hint_np[i], mode='RGBA') for i in range(B)]

    return edge_map_images, color_hint_images

def target_to_pil_images (targets) : 
    targets = torch.clamp((targets / 2) + 0.5, 0.0, 1.0)

    targets = torch.clamp((targets * 255), 0.0, 255.0).to(torch.uint8)
    targets = targets.detach().cpu().numpy() 

    B, *_ = targets.shape

    target_imgs = [Image.fromarray(targets[i], mode='RGB') for i in range(B)]
    return target_imgs

def captions_to_pil_images (captions) :
    return [txt_as_pil((512, 512), c, size=16) for c in captions]

def batch_to_pil (batch) : 
    edge_map_imgs, color_hint_imgs = hint_to_pil_images(batch['hint'])
    target_imgs = target_to_pil_images(batch['jpg'])
    caption_imgs = captions_to_pil_images(batch['txt'])
    imgs = [target_imgs, caption_imgs, edge_map_imgs, color_hint_imgs]
    img = make_image_grid(imgs, img_type='RGBA')
    return img

def find_first_true (arr) :
    """ A numpy array of boolean values of shape (N,) """
    indices = np.where(arr)[0]
    return indices[0] if indices.size > 0 else None

def find_first_false (arr) :
    """ A numpy array of boolean values of shape (N,) """
    indices = np.where(~arr)[0]
    return indices[0] if indices.size > 0 else None

def extract_edge_map(pil_img, target_size=128) : 
    # optionally resize image in the beginning
    resize = random.choice([0.99, 0.9, 0.8, 0.75, 0.5, None])
    if resize is not None :
        resize = int(target_size * resize)
        pil_img = aspectRatioPreservingResizePIL(pil_img, resize)

    # convert image to gray
    np_img = np.array(pil_img)
    gray_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur with some probability
    apply_gaussian_blur = random.choice([True, False]) 
    if apply_gaussian_blur :
        kernel_size = random.choice([3, 5]) 
        sigmaX = random.choice([1, 3, 5])
        gray_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), sigmaX)

    # apply canny
    low_thresh = random.randint(50, 150)
    high_thresh = max(low_thresh + 10, min(255, random.randint(2 * low_thresh, 5 * low_thresh)))
    canny_img = cv2.Canny(gray_img, low_thresh, high_thresh)

    # apply dilation with some probability
    apply_dilation = random.choice([False])
    if apply_dilation :
        dilation_kernel_size = random.choice([2, 4])
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        canny_img = cv2.dilate(canny_img, kernel, iterations=1)

    # flip black and white
    canny_img = 255 - canny_img
    canny_img_pil = Image.fromarray(canny_img)

    # optionally downsample and upsample
    smaller_dim = min(canny_img.shape)
    apply_downsample = random.choice([None])
    if apply_downsample is not None :
        new_dim = int(apply_downsample * smaller_dim)
        canny_img_pil = aspectRatioPreservingResizePIL(canny_img_pil, new_dim)
        canny_img_pil = aspectRatioPreservingResizePIL(canny_img_pil, smaller_dim)

    # resize edge map to target size and return it
    canny_img_pil = aspectRatioPreservingResizePIL(canny_img_pil, target_size)
    return canny_img_pil

def add_point_color_strokes (color_strokes, np_img): 
    h, w = np_img.shape[:2]
    
    thickness = random.choice([1, 2, 4, 8])
    st_y, st_x = random.randint(thickness, h - thickness), random.randint(thickness, w - thickness)
    
    color_strokes[st_y-thickness:st_y+thickness, st_x-thickness:st_x+thickness, 3] = 255
    color = np_img[st_y-thickness:st_y+thickness, st_x-thickness:st_x+thickness].mean((0, 1)).astype(np.uint8)
    color_strokes[st_y-thickness:st_y+thickness, st_x-thickness:st_x+thickness, :3] = color

def add_color_scribbles (color_strokes, np_img): 
    h, w = np_img.shape[:2]
    
    thickness = random.choice([1, 2, 4])
    st_y, st_x = random.randint(thickness, h - thickness), random.randint(thickness, w - thickness)

    color = np_img[st_y-thickness:st_y+thickness, st_x-thickness:st_x+thickness].mean((0, 1)).astype(np.uint8)

    color_strokes[st_y-thickness:st_y+thickness, st_x-thickness:st_x+thickness, 3] = 255
    color_strokes[st_y-thickness:st_y+thickness, st_x-thickness:st_x+thickness, :3] = color

    while True : 
        # generate dir
        theta = random.choice(np.linspace(0, 2 * np.pi))
        vy, vx = np.sin(theta), np.cos(theta)
        # generate step size in pixels
        step_size = random.randint(10, 20)
        # keep walking in step size till color doesn't change too much
        st_y_f, st_x_f = st_y, st_x
        for i in range(step_size): 
            st_y_f, st_x_f = st_y_f + vy, st_x_f + vx
            st_y, st_x = round(st_y_f), round(st_x_f)
            try : 

                yl = clamp(st_y - thickness, 0, h - 1)
                yh = clamp(st_y + thickness, 0, h - 1)

                xl = clamp(st_x - thickness, 0, w - 1)
                xh = clamp(st_x + thickness, 0, w - 1)
                if yl >= yh or xl >= xh : 
                    break

                new_color = np_img[yl:yh, xl:xh].mean((0, 1)).astype(np.uint8)
                color_strokes[yl:yh, xl:xh, 3] = 255
                color_strokes[yl:yh, xl:xh, :3] = color
            except Exception as e :
                # probably went out of bounds
                break
        # pick a new direction
        theta = np.random.vonmises(theta, 0.5)
        if random.random() < 0.2 :
            break

def add_random_walk (color_strokes, np_img) : 
    # get shape of image, starting point and length of walk
    h, w = np_img.shape[:2]
    st_y, st_x = random.randint(0, h - 1), random.randint(0, w - 1)
    L = random.randint(min(h, w), h * w) 

    # construct walk
    dirs = np.array([[-1, 0], [0, -1], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=int)
    rng_idx = np.random.randint(0, 8, (L,))
    steps = dirs[rng_idx]
    px_points = np.cumsum(steps, axis=0)
    px_points[:, 0] += st_y
    px_points[:, 1] += st_x

    # find when walk goes out of bounds and truncate it
    y_mask = (px_points[:, 0] < h) & (px_points[:, 0] >= 0)
    x_mask = (px_points[:, 1] < w) & (px_points[:, 1] >= 0)
    ff_id = find_first_false(y_mask & x_mask)
    px_points = px_points[:ff_id]
    
    # create mask from walk
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[px_points[:, 0], px_points[:, 1]] = 255

    # optionally, dilate the walk
    thickness = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
    if thickness > 0 : 
        footprint = np.ones((thickness, thickness), dtype=np.uint8) 
        mask = skimage.filters.rank.maximum(mask, footprint)

    # copy over colors from input image
    color_strokes[mask > 0, :3] = np_img[mask > 0]
    color_strokes[mask > 0, 3] = mask[mask > 0]
        
def extract_color_hint (pil_img, hint_adder, target_size=512) : 
    # (optionally) blur the image by downsampling
    ds = np.linspace(0.33, 0.99, 40).tolist() + [None]
    apply_downsample = random.choice(ds)
    if apply_downsample is not None: 
        new_dim = int(apply_downsample * min(pil_img.size))
        pil_img = aspectRatioPreservingResizePIL(pil_img, new_dim)

    # get dimensions of image
    np_img = np.array(pil_img)
    h, w = np_img.shape[:2]
    
    # figure out how many strokes to add
    n_strokes = random.choice(list(range(3, 30)))

    # employ hint_adder to add color hints
    color_hint = np.zeros((h, w, 4), dtype=np.uint8)
    for _ in range(n_strokes) : 
        hint_adder(color_hint, np_img)
    color_hint_pil = Image.fromarray(color_hint, 'RGBA')

    # optionally downsample and upsample
    smaller_dim = min(h, w)
    apply_downsample = random.choice(ds)
    if apply_downsample is not None :
        new_dim = int(apply_downsample * smaller_dim)
        color_hint_pil = aspectRatioPreservingResizePIL(color_hint_pil, new_dim)
        color_hint_pil = aspectRatioPreservingResizePIL(color_hint_pil, smaller_dim)

    # resize edge map to target size and return it
    color_hint_pil = aspectRatioPreservingResizePIL(color_hint_pil, target_size)
    return color_hint_pil
    
def color_hints_and_outline_extractor (pil_img, target_size=512) : 
    """
    Returns the hint [H, W, 5] that is normalized to be in range [0, 1]
    """
    # get edge map
    edge_map = extract_edge_map(pil_img, target_size=target_size) # PIL
    edge_map_np = np.array(edge_map) # [H, W] 

    # get color hint
    hint_adder = random.choice([add_random_walk]) # removed color scribbles
    color_hint = extract_color_hint(pil_img, hint_adder, target_size=target_size)
    color_hint_np = np.array(color_hint) # [H, W, 4]

    H, W = edge_map_np.shape

    # combine the two 
    hint = np.concatenate((edge_map_np.reshape(H, W, 1), color_hint_np), axis=2) # [H, W, 5], np.uint8 hopefully!
    hint = hint.astype(np.float32) / 255.0
    return hint # [H, W, 5], [0, 1]

def load_dataframe (path_or_dir) : 
    if osp.isdir(path_or_dir) : 
        paths = glob(f'{path_or_dir}/*')
        dfs = [] 
        print(f'Loading dataframes from {len(paths)} files')
        for path in tqdm(paths) : 
            ext = osp.splitext(path)[1]
            if ext == '.parquet' : 
                dfs.append(pd.read_parquet(path))
            elif ext in ['.csv', '.tsv'] : 
                dfs.append(pd.read_csv(path))
            else : 
                raise ValueError(f'Don\'t know how to load {path}')
        return combine_dataframes(dfs)
    else : 
        ext = osp.splitext(path_or_dir)[1]
        if ext == '.parquet' : 
            return pd.read_parquet(path_or_dir)
        elif ext in ['.csv', '.tsv'] : 
            return pd.read_csv(path_or_dir)
        else :
            raise ValueError(f'Don\'t know how to load {path_or_dir}')

def combine_dataframes(df_list):
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def target_to_pil_images (targets) : 
    targets = torch.clamp((targets / 2) + 0.5, 0.0, 1.0)

    targets = torch.clamp((targets * 255), 0.0, 255.0).to(torch.uint8)
    targets = targets.detach().cpu().numpy() 

    B, *_ = targets.shape

    target_imgs = [Image.fromarray(targets[i], mode='RGB') for i in range(B)]
    return target_imgs

def star_sanitizer(txt) :
    match = re.search(r'\*\*(.*?)\*\*', txt)
    return match.group(1) if match else txt

def url_sanitizer (txt) : 
    return re.sub(r'<.*?>', '', txt)

def args_sanitizer (txt) :
    return re.sub(r'\s*--\w+\s+[\w\.]+', '', txt)

def midjourney_txt_sanitizer (txt) : 
    txt = star_sanitizer(txt)
    txt = url_sanitizer(txt)
    txt = args_sanitizer(txt)
    return txt.strip()

class DataframeDataset(Dataset):
    def __init__(self, dataframe, text_sanitizer=None, annotator=None, urlKey='url', widthKey='width', heightKey='height', captionKey='caption', target_size=512, split='train'):
        self.dataframe = dataframe
        self.annotator = annotator
        self.text_sanitizer = text_sanitizer
        self.urlKey = urlKey
        self.widthKey = widthKey
        self.heightKey = heightKey
        self.captionKey = captionKey
        self.target_size = target_size

        # Filter dataframe
        self.dataframe = self.dataframe[
            (self.dataframe[self.widthKey] >= 0) & 
            (self.dataframe[self.heightKey] >= 0) & 
            (self.dataframe[self.widthKey] / self.dataframe[self.heightKey] >= 0.9) &
            (self.dataframe[self.widthKey] / self.dataframe[self.heightKey] <= 1.1) &
            (~self.dataframe[self.captionKey].str.contains('Variations', na=False)) &
            (~self.dataframe[self.captionKey].str.contains('\(fast\)', na=False)) &
            (~self.dataframe[self.captionKey].str.contains('\(relaxed\)', na=False))
        ]

        N = len(self.dataframe)
        I = int(N * 0.95)
        if split == 'train': 
            self.dataframe = self.dataframe.head(I)
        else :
            self.dataframe = self.dataframe.tail(N - I)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try: 
            row = self.dataframe.iloc[idx]
            url = row[self.urlKey]
            caption = row[self.captionKey]

            # Fetch image
            response = requests.get(url, timeout=2)
            img = Image.open(BytesIO(response.content)).convert('RGB')

            # Resize and crop
            img = aspectRatioPreservingResizePIL(img, self.target_size)
            transform = transforms.CenterCrop(self.target_size)
            img = transform(img)
            target = np.array(img)
            target = (target.astype(np.float32) / 127.5) - 1.0
            if self.text_sanitizer is not None :
                caption = self.text_sanitizer(caption)

            if random.random() < 0.2 : 
                caption = ''

            # Annotate
            if self.annotator is not None :
                source = self.annotator(img)
                return {'jpg': target, 'txt': caption, 'hint': source}
            return {'jpg': target, 'txt': caption }
        except Exception as e: 
            print(e)
            return self.__getitem__(random.randint(0, len(self) - 1))

class ColorDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, target_size=512, num_workers=10):
        super().__init__()
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # load all pieces.
        mj_dataset_train = DataframeDataset(
            load_dataframe(MIDJOURNEY_MSGS_PARQUET_FILE), 
            text_sanitizer=midjourney_txt_sanitizer, 
            captionKey='content',
            split='train',
            target_size=self.target_size,
            annotator=color_hints_and_outline_extractor
        )
        mj_dataset_val = DataframeDataset(
            load_dataframe(MIDJOURNEY_MSGS_PARQUET_FILE), 
            text_sanitizer=midjourney_txt_sanitizer, 
            captionKey='content',
            split='val',
            target_size=self.target_size,
            annotator=color_hints_and_outline_extractor
        )
        print("Loaded Midjourney Messages") 
        cc_dataset_train = DataframeDataset(
            load_dataframe(CONCEPTUAL_CAPTIONS_CSV_FILE), 
            split='train',
            target_size=self.target_size,
            annotator=color_hints_and_outline_extractor
        )
        cc_dataset_val = DataframeDataset(
            load_dataframe(CONCEPTUAL_CAPTIONS_CSV_FILE), 
            split='val',
            target_size=self.target_size,
            annotator=color_hints_and_outline_extractor
        )
        print("Loaded Conceptual Captions") 
        self.dataset_train = ConcatDataset([mj_dataset_train, cc_dataset_train])
        self.dataset_val = ConcatDataset([mj_dataset_val, cc_dataset_val])

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

if  __name__ == "__main__" :
    datamodule = ColorDataModule(batch_size=1, num_workers=0)
    datamodule.setup()
    print(f'Size of training dataset = {len(datamodule.dataset_train)}')
    print(f'Size of validation dataset = {len(datamodule.dataset_val)}')
    dataloader = datamodule.train_dataloader()
    for i, batch in enumerate(tqdm(dataloader)) : 
        img = batch_to_pil(batch)
        # if i < 20 :
        #     img.save(f'batch_{i}.png')
        if i > 100 :
            break
    print(batch.keys())
