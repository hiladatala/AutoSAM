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
            self.imgs_root = os.path.join(self.root, 'Test', 'img')
            self.masks_root = os.path.join(self.root, 'Test', 'mask')
        self.paths = os.listdir(self.imgs_root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.loops = loops
        self.sam_trans = sam_trans
        self.all_slices = self.preload_all_slices()
        print('num of data:{}'.format(len(self.paths)))
        print(f'Number of slices in the dataset: {len(self.all_slices)}')

    def preload_all_slices(self):
        all_slices = []
        for file_path in self.paths:
            # Construct the full path to the image and mask files
            mask_path = file_path.split('.')[0] + '.nii.gz'
            img = self.loader(os.path.join(self.imgs_root, file_path), is_mask=False)
            mask = self.loader(os.path.join(self.masks_root, mask_path), is_mask=True)

            # Resize the image and mask to 256x256x120
            img = zoom(img, (256 / img.shape[0], 256 / img.shape[1], 96 / img.shape[2]))
            mask = zoom(mask, (256 / mask.shape[0], 256 / mask.shape[1], 96 / mask.shape[2]))
            mask = (mask == 6)

            # Loop over all slices and store them in the all_slices list
            #random.seed(42)
            #random_array = [random.randint(0, 91) for _ in range(92)]
            #selected_numbers = random.sample(random_array, 30)

            random_array = [random.randint(0, 64) for _ in range(64)]
            idx = random.choice(random_array)


            for i in range(30):
                #idx = selected_numbers[i]
                #img_slice = img[:, :, idx]  # Get a specific slice
                #mask_slice = mask[:, :, idx]  # Get the corresponding mask slice

                img_slice = img[:, :, idx + i]
                img_slice = (img_slice - np.mean(img_slice)) / np.std(img_slice)

                mask_slice = mask[:, :, idx + i]

                # Convert to 3-channel (RGB) image
                img_slice = np.stack([img_slice*1/3] * 3, axis=-1)  # (256, 256, 3)


                # Create a figure with two subplots (one for image, one for mask)
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Display the image slice
                axes[0].imshow(img_slice, cmap="gray")
                axes[0].set_title("CT Scan Slice")
                axes[0].axis("off")  # Hide axes

                # Display the mask slice
                # print(np.unique(mask_slice))
                # mask_slice = np.where(mask_slice > 0.1, 1, 0).astype(np.float32)

                axes[1].imshow(mask_slice, cmap="gray")
                axes[1].set_title("Segmentation Mask Slice")
                axes[1].axis("off")  # Hide axes

                # Show the plot
                plt.tight_layout()
                plt.show()


                # Apply transformations
                img_slice, mask_slice = self.transform(img_slice, mask_slice)
                original_size = tuple(img_slice.shape[1:3])  # (256, 256)
                img_slice, mask_slice = self.sam_trans.apply_image_torch(img_slice), self.sam_trans.apply_image_torch(
                    mask_slice)

                # Process mask to binary format
                mask_slice[mask_slice > 0.1] = 1
                mask_slice[mask_slice <= 0.1] = 0
                image_size = tuple(img_slice.shape[1:3])  # (256, 256)

                # Store the processed slice and metadata as a tuple
                all_slices.append((self.sam_trans.preprocess(img_slice),
                                   self.sam_trans.preprocess(mask_slice),
                                   torch.Tensor(original_size),
                                   torch.Tensor(image_size)))

        return all_slices



    def __getitem__(self, index):
        return self.all_slices[index % len(self.all_slices)]



        '''
        index = index % len(self.paths)
        file_path = self.paths[index]
        mask_path = file_path.split('.')[0] + '.nii.gz'
        img = self.loader(os.path.join(self.imgs_root, file_path), is_mask=False)
        mask = self.loader(os.path.join(self.masks_root, mask_path), is_mask=True)

        img = zoom(img, (256 / img.shape[0], 256 / img.shape[1], 120 / img.shape[2]))
        mask = zoom(mask, (256 / mask.shape[0], 256 / mask.shape[1], 120 / mask.shape[2]))


        dataset_items = []
        img  = np.stack([img[:,:,0]] * 3, axis=-1)
        mask = mask[:,:,0]
        img, mask = self.transform(img, mask)
        original_size = tuple(img.shape[1:3])
        img, mask = self.sam_trans.apply_image_torch(img), self.sam_trans.apply_image_torch(mask)
        print(img.shape)
        print(mask.shape)
        mask[mask > 0.1] = 1
        mask[mask <= 0.1] = 0
        image_size = tuple(img.shape[1:3])
        return self.sam_trans.preprocess(img), self.sam_trans.preprocess(mask), torch.Tensor(
            original_size), torch.Tensor(image_size)
        '''

    def __len__(self):
        return len(self.paths) * self.loops


def get_lung_dataset(args, sam_trans):
    datadir = 'C:\Hila\Registration task\Dataset\Lung'
    transform_train, transform_test = get_lung_transform(args)
    ds_train = ImageLoader(datadir, train=True, transform=transform_train, sam_trans=sam_trans, loops=5)
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
