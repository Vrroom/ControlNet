import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import random
import skimage
import warnings
from ldm.util import log_txt_as_img
from imageOps import *

warnings.filterwarnings('error', message='Corrupt EXIF data.*')

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
    resize = random.choice([128, 256, 512, None])
    if resize is not None :
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
    low_thresh = random.randint(50, 100)
    high_thresh = min(255, random.randint(2 * low_thresh, 3 * low_thresh))
    canny_img = cv2.Canny(gray_img, low_thresh, high_thresh)

    # apply dilation with some probability
    apply_dilation = random.choice([True, False])
    if apply_dilation :
        dilation_kernel_size = random.choice([2, 4])
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        canny_img = cv2.dilate(canny_img, kernel, iterations=1)

    # flip black and white
    canny_img = 255 - canny_img
    canny_img_pil = Image.fromarray(canny_img)

    # optionally downsample and upsample
    smaller_dim = min(canny_img.shape)
    apply_downsample = random.choice([0.33, 0.5, 0.75, 0.8, None])
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
    L = 200 # random.randint(20, 100) 

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
    thickness = random.choice([0, 1, 2, 4])
    if thickness > 0 : 
        footprint = np.ones((thickness, thickness), dtype=np.uint8) 
        mask = skimage.filters.rank.maximum(mask, footprint)

    # copy over colors from input image
    color_strokes[mask > 0, :3] = np_img[mask > 0]
    color_strokes[mask > 0, 3] = mask[mask > 0]
        
def extract_color_hint (pil_img, hint_adder, target_size=512) : 
    # (optionally) blur the image by downsampling
    apply_downsample = random.choice([0.33, 0.5, 0.75, 0.8, None])
    if apply_downsample is not None: 
        new_dim = int(apply_downsample * min(pil_img.size))
        pil_img = aspectRatioPreservingResizePIL(pil_img, new_dim)

    # get dimensions of image
    np_img = np.array(pil_img)
    h, w = np_img.shape[:2]
    
    # figure out how many strokes to add
    n_strokes = random.choice([2 ** i for i in range(8)])

    # employ hint_adder to add color hints
    color_hint = np.zeros((h, w, 4), dtype=np.uint8)
    for _ in range(n_strokes) : 
        hint_adder(color_hint, np_img)
    color_hint_pil = Image.fromarray(color_hint, 'RGBA')

    # optionally downsample and upsample
    smaller_dim = min(h, w)
    apply_downsample = random.choice([0.33, 0.5, 0.75, 0.8, None])
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
    hint_adder = random.choice([add_point_color_strokes, add_random_walk]) # removed color scribbles
    color_hint = extract_color_hint(pil_img, hint_adder, target_size=target_size)
    color_hint_np = np.array(color_hint) # [H, W, 4]

    H, W = edge_map_np.shape

    # combine the two 
    hint = np.concatenate((edge_map_np.reshape(H, W, 1), color_hint_np), axis=2) # [H, W, 5], np.uint8 hopefully!
    hint = hint.astype(np.float32) / 255.0
    return hint # [H, W, 5], [0, 1]

class DanbooruColorDataset(Dataset):
    def __init__(self, target_size=512, split='train'):
        with open('./training/danbooru_color/captions.json') as fp:
            self.captions = json.load(fp)
        self.image_paths = list(self.captions.keys())
        self.target_size = target_size
        I = int(len(self.image_paths) * 0.95)
        if split == 'train': 
            self.image_paths = self.image_paths[:I]
        else :
            self.image_paths = self.image_paths[I:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try : 
            image_path = self.image_paths[idx]
            target_pil = Image.open(image_path).convert('RGB').resize((self.target_size, self.target_size))
            target = np.array(target_pil)
            target = (target.astype(np.float32) / 127.5) - 1.0
            prompt = random.choice(self.captions[image_path])
            source = color_hints_and_outline_extractor(target_pil, self.target_size)
            print(source.min(), source.max(), target.min(), target.max())
            return dict(jpg=target, txt=prompt, hint=source)
        except Exception : 
            print(1)
            # if there is an error, just go to next id. prob of error < 0.00001
            return self.__getitem__((idx + 1) % self.__len__())

class DanbooruColorDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, target_size=512, num_workers=10):
        super().__init__()
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.danbooru_train = DanbooruColorDataset(target_size=self.target_size, split='train')
        self.danbooru_val = DanbooruColorDataset(target_size=self.target_size, split='val')

    def train_dataloader(self):
        return DataLoader(self.danbooru_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.danbooru_val, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__" : 
    dm = DanbooruColorDataModule(batch_size=30, num_workers=10) 
    dm.setup()
    dataloader = dm.train_dataloader()
    for i, batch in enumerate(tqdm(dataloader)) : 
        img = batch_to_pil(batch)
        # img.save(f'batch_{i}.png')
        if i > 0 :
            break
