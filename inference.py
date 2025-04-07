import torch.utils.data
import torch
import os
from models.model_single import ModelEmb
from dataset.glas import get_glas_dataset
from dataset.MoNuBrain import get_monu_dataset
from dataset.polyp import get_polyp_dataset, get_tests_polyp_dataset
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from train import get_input_dict, norm_batch, get_dice_ji
import cv2
from dataset.LungData import get_lung_dataset
import matplotlib.pyplot as plt
from dataset.tfs import get_lung_transform

sam_args = {
    'sam_checkpoint': "/media/cilab/DATA/Hila/Data/Projects/AutoSAM/sam_vit_h.pth",
    'model_type': "vit_h",
    'generator_args':{
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


def inference_ds(ds, model, sam, transform, epoch, args):
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])
    transform_train, transform_test = get_lung_transform(args)
    count = 0
    for imgs, gts, original_sz, img_sz in pbar:
        count = count + 1
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)
        prediction_masks = torch.zeros(NumSliceDim, Idim, Idim)
        #og_masks = torch.zeros(NumSliceDim, Idim, Idim)

        for slice in range(NumSliceDim):
            orig_imgs_slice = orig_imgs[:, :, :, slice].unsqueeze(0)
            orig_imgs_slice_small = F.interpolate(orig_imgs_slice, (Idim, Idim), mode='bilinear', align_corners=True)
            orig_imgs_slice_small = orig_imgs_slice_small * 1 / 3
            orig_imgs_slice_small = orig_imgs_slice_small.repeat(1, 3, 1, 1)

            ct_slice = orig_imgs_slice_small.squeeze()
            ct_slice = ct_slice.permute(1, 2, 0)
            mask_slice = gts[:, :, :, slice].squeeze()

            dense_embeddings = model(orig_imgs_slice_small)

            ct_slice, mask_slice = transform_test(ct_slice.cpu(), mask_slice.cpu())
            original_sz = ct_slice.shape[1:3]
            ct_slice = transform.apply_image_torch(ct_slice)
            ct_slice = transform.preprocess(ct_slice).cuda()
            img_sz = ct_slice.shape[1:3]
            img_sz = torch.tensor(img_sz).unsqueeze(0)
            original_sz = torch.tensor(original_sz).unsqueeze(0)

            batched_input = get_input_dict([ct_slice], [original_sz], [img_sz])
            masks_pred = norm_batch(sam_call(batched_input, sam, dense_embeddings))

            input_size = tuple([int(x) for x in img_sz[0].squeeze().tolist()])
            original_size = tuple([int(x) for x in original_sz[0].squeeze().tolist()])
            masks_pred = sam.postprocess_masks(masks_pred, input_size=input_size, original_size=original_size)
            mask_slice = sam.postprocess_masks(mask_slice.unsqueeze(0).unsqueeze(1), input_size=input_size,
                                               original_size=original_size)
            masks_pred = F.interpolate(masks_pred, (Idim, Idim), mode='bilinear', align_corners=True)
            mask_slice = F.interpolate(mask_slice, (Idim, Idim), mode='nearest')
            prediction_masks[slice, :, :] = masks_pred.squeeze()
            #og_masks[slice, :, :] = mask_slice.squeeze()

        prediction_masks[prediction_masks > 0.5] = 1
        prediction_masks[prediction_masks <= 0.5] = 0
        prediction_masks = prediction_masks.permute(1, 2, 0)

        prediction_masks = prediction_masks.detach().cpu().numpy()
        gts_np = gts.squeeze().detach().cpu().numpy()
        scans = orig_imgs.squeeze().squeeze()
        scans = scans.cpu().numpy()

        # Create a figure with two subplots (1 row, 2 columns)
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        scan = scans[:, :, 35]
        scan = (scan - scan.min()) / (scan.max() - scan.min())
        axes[0].imshow(scan, cmap="gray")
        axes[0].set_title(f"Test set scan number {count} - slice 35")
        axes[0].axis("off")  # Hide axes

        axes[1].imshow(gts_np[:, :, 35], cmap="gray")
        axes[1].set_title("Ground Truth mask")
        axes[1].axis("off")  # Hide axes

        axes[2].imshow(prediction_masks[:, :, 35], cmap="gray")
        axes[2].set_title("Predicted mask")
        axes[2].axis("off")  # Hide axes
        plt.show()

        dice, ji = get_dice_ji(prediction_masks,
                               gts.squeeze().detach().cpu().numpy())

        print(f"Dice of scan {count} slice 35: {dice}")
        print(f"IoU of scan {count} slice 35: {ji}")

        iou_list.append(ji)
        dice_list.append(dice)
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: IoU {iou:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice=np.mean(dice_list),
                iou=np.mean(iou_list)))
    model.train()
    return np.mean(iou_list)


def sam_call(batched_input, sam, dense_embeddings):
    input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
    image_embeddings = sam.image_encoder(input_images)
    sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks



def main(args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelEmb(args=args).cuda()
    model1 = torch.load(args['path_best'], map_location=device, weights_only=False)
    model.load_state_dict(model1.state_dict())
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    if args['task'] == 'monu':
        trainset, testset = get_monu_dataset(args, sam_trans=transform)
    elif args['task'] == 'glas':
        trainset, testset = get_glas_dataset(sam_trans=transform)
    elif args['task'] == 'polyp':
        trainset, testset = get_polyp_dataset(args, sam_trans=transform)
    elif args['task'] == 'lung':
        trainset, testset = get_lung_dataset(args, sam_trans=transform)

    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)
    with torch.no_grad():
        model.eval()
        inference_ds(ds_val, model.eval(), sam, transform, 0, args)


if __name__ == '__main__':
    # glas 29 256 h
    # monu 34 512 h
    # polyp 56 352 b

    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='lung', help='evaluation iteration', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-folder', '--folder', default=313, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=256, help='image size', required=False)
    parser.add_argument('-NumSliceDim', '--NumSliceDim', default=64, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    args = vars(parser.parse_args())
    base_save_dir = "/media/cilab/DATA/Hila/Projects/AutoSAM/results"
    results_dir = os.path.join(base_save_dir, f'gpu{args["folder"]}')
    args['path_best'] = os.path.join(base_save_dir,
                                     'gpu' + str(args['folder']),
                                     'net_best.pth')
    args['vis_folder'] = os.path.join(base_save_dir, 'gpu' + str(args['folder']), 'vis')
    os.makedirs(args['vis_folder'], exist_ok=True)
    main(args=args)

