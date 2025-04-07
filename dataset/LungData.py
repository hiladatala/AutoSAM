import torch
import PIL
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision.datasets as tvdataset
from dataset.tfs import get_lung_transform
import cv2
import nibabel as nib
from scipy.ndimage import zoom
from scipy.ndimage import label
import matplotlib.pyplot as plt
import random
import pickle
import os
import gc
from tqdm import tqdm


def cv2_loader(path, is_mask):
    if is_mask:
        img = cv2.imread(path, 0)
        img[img > 0] = 1
    else:
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img

def nib_loader(path, is_mask):
    if is_mask:
        img = nib.load(path).get_fdata()
        img = np.where(img > 0.1, 1, 0).astype(np.float32)
    else:
        img = nib.load(path).get_fdata()
    return img


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=nib_loader,
                 sam_trans=None, loops=1):
        self.root = root
        if train:
            self.imgs_root = os.path.join(self.root, 'Training', 'img')
            self.masks_root = os.path.join(self.root, 'Training', 'mask')
        else:
            self.imgs_root = os.path.join(self.root, 'Testing', 'img')
            self.masks_root = os.path.join(self.root, 'Testing', 'mask')
        self.paths = os.listdir(self.imgs_root)
        self.mask_paths = os.listdir(self.masks_root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.loops = loops
        self.sam_trans = sam_trans
        self.all_volumes = self.preload_all_volumes_as_is(train=train)
        print('num of data:{}'.format(len(self.paths)))
        print(f'Number of slices in the dataset: {len(self.all_volumes)}')

    def preload_all_volumes_as_is(self, downscale_factor=(0.5, 0.5, 1.0), train=False):

        cache_dir = '/media/cilab/DATA/Hila/Data/Projects/AutoSAM'
        os.makedirs(cache_dir, exist_ok=True)
        all_volumes = []

        print("Loading full volumes and saving (with existence check):")
        for file_idx, file_path in enumerate(tqdm(self.paths, desc="Loading full volumes")):
            volume_pt_path = os.path.join(cache_dir, f"volume_{file_idx}.pt")

            # ✅ Check if .pt file already exists
            if os.path.exists(volume_pt_path):
                print(f"⚡ Volume {file_idx} already cached. Skipping.")
                volume_data = torch.load(volume_pt_path)
                all_volumes.append(volume_data)
                continue

            # Load and process
            img = self.loader(os.path.join(self.imgs_root, file_path), is_mask=False).astype(np.float32)
            mask = self.loader(os.path.join(self.masks_root, self.mask_paths[file_idx]), is_mask=False).astype( np.float32)
            mask = (mask == 6)
            mask = mask.astype(float)

            img = zoom(img, (256 / img.shape[0], 256 / img.shape[1], 64 / img.shape[2]))
            mask = zoom(mask, (256 / mask.shape[0], 256 / mask.shape[1], 64 / mask.shape[2]))

            '''
            plt.imshow(mask[:, :, 30])
            plt.show()

            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # Display the image slice
            axes[0].imshow(img[:, :, 30], cmap="gray")
            axes[0].set_title("CT Scan Slice")
            axes[0].axis("off")  # Hide axes

            axes[1].imshow(mask[:, :, 30], cmap="gray")
            axes[1].set_title("Segmentation Mask Slice")
            axes[1].axis("off")  # Hide axes
            plt.show()
            '''

            img_tensor = torch.tensor(img, dtype=torch.float32)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            original_size = img.shape[0:2]
            image_size = img.shape[0:2]

            # Save .pt file
            torch.save((img_tensor, mask_tensor, original_size, image_size), volume_pt_path)
            print(f"💾 Saved volume tensor at: {volume_pt_path}")

            all_volumes.append((img_tensor, mask_tensor, original_size, image_size))

        print(f"✅ Done! Total volumes processed or loaded from cache: {len(all_volumes)}")
        return all_volumes


    def __getitem__(self, index):
        img_tensor, mask_tensor, original_size, img_size = self.all_volumes[index % len(self.all_volumes)]
        return img_tensor, mask_tensor, original_size, img_size

    def __len__(self):
        return len(self.paths) * self.loops


def get_lung_dataset(args, sam_trans):
    datadir = '/media/cilab/DATA/Hila/Data/Projects/AutoSAM/Abdomen'
    transform_train, transform_test = get_lung_transform(args)
    ds_train = ImageLoader(datadir, train=True, transform=transform_train, sam_trans=sam_trans, loops=1)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test, sam_trans=sam_trans)
    return ds_train, ds_test


if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    import os
    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
    from segment_anything.utils.transforms import ResizeLongestSide

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-Idim', '--Idim', default=512, help='learning_rate', required=False)
    parser.add_argument('-pSize', '--pSize', default=4, help='learning_rate', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='learning_rate', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='learning_rate', required=False)
    parser.add_argument('-rotate', '--rotate', default=20, help='learning_rate', required=False)
    args = vars(parser.parse_args())

    sam_args = {
        'sam_checkpoint': "../cp/sam_vit_b.pth",
        'model_type': "vit_b",
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': 0,
    }
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    sam_trans = ResizeLongestSide(sam.image_encoder.img_size)
    ds_train, ds_test = get_lung_dataset(args, sam_trans)
    ds = torch.utils.data.DataLoader(ds_train,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=True,
                                     drop_last=True)
    pbar = tqdm(ds)
    mean0_list = []
    mean1_list = []
    mean2_list = []
    std0_list = []
    std1_list = []
    std2_list = []
    for i, (img, mask, _, _) in enumerate(pbar):
        a = img.mean(dim=(0, 2, 3))
        b = img.std(dim=(0, 2, 3))
        mean0_list.append(a[0].item())
        mean1_list.append(a[1].item())
        mean2_list.append(a[2].item())
        std0_list.append(b[0].item())
        std1_list.append(b[1].item())
        std2_list.append(b[2].item())
    print(np.mean(mean0_list))
    print(np.mean(mean1_list))
    print(np.mean(mean2_list))

    print(np.mean(std0_list))
    print(np.mean(std1_list))
    print(np.mean(std2_list))

        # a = img.squeeze().permute(1, 2, 0).cpu().numpy()
        # b = mask.squeeze().cpu().numpy()
        # a = (a - a.min()) / (a.max() - a.min())
        # cv2.imwrite('kaki.jpg', 255*a)
        # cv2.imwrite('kaki_mask.jpg', 255*b)
