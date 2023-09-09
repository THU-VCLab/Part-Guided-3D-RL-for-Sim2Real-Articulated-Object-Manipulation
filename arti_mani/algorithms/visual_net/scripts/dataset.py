import os
import random
import time

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from arti_mani import KPTDATA_DIR, SEGDATA_DIR

# from copy_paste import CopyPaste
from segmentation_models_pytorch import utils as smp_utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

IMAGE_HEIGHT = 144
IMAGE_WIDTH = 256


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    plt.figure(figsize=(16, 16))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def vis_keypoints(rgb, depth, keypoints, color=(255, 0, 0), diameter=2):
    plot_rgb = rgb.copy()

    for x, y in keypoints[:, :2]:
        cv2.circle(plot_rgb, (round(x), round(y)), diameter, color, -1)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.subplot(1, 3, 2)
    plt.imshow(depth)
    plt.subplot(1, 3, 3)
    plt.imshow(plot_rgb)
    plt.show()


def spatial_augmentation(copy_paste=False):
    transforms = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomSizedCrop(
            (int(IMAGE_HEIGHT * 0.85), IMAGE_HEIGHT),
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            w2h_ratio=IMAGE_WIDTH / IMAGE_HEIGHT,
            p=1,
        ),
    ]
    # if copy_paste:
    #     transforms = [CopyPaste(blend=False, p=0.2)] + transforms
    return albu.Compose(transforms)


def rgb_pixel_augmentation():
    transforms = [
        albu.GaussNoise(var_limit=(10, 50), p=0.5),
        albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    ]
    return albu.Compose(transforms)


def depth_pixel_augmention():
    train_transform = [
        albu.GaussNoise(var_limit=0.03, p=0.5),
        AddSaltPepperNoise(density=0.05, p=0.5),
    ]
    return albu.Compose(train_transform)


class AddSaltPepperNoise(ImageOnlyTransform):
    def __init__(self, density=0.0, always_apply=False, p=0.5):
        super(AddSaltPepperNoise, self).__init__(always_apply, p)
        self.density = density
        self.p = p

    def apply(self, img, mask=None, **params):
        if random.uniform(0, 1) < self.p:
            img[mask == 0] = 0  # 椒
            img[mask == 1] = 2  # 盐
            return img
        else:
            return img

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        Nd = self.density
        Sd = 1 - Nd
        h, w, c = image.shape
        mask = np.random.choice(
            (0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd]
        )  # h,w,q
        mask = np.repeat(mask, c, axis=2)  # h,w,c
        return {"mask": mask}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "density"


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32."""
    test_transform = [
        albu.PadIfNeeded(IMAGE_HEIGHT, IMAGE_WIDTH)
        # albu.PadIfNeeded(80, 128)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform.

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class SegDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        data_dir (str): path to RGBD data and seg masks
        class_values (list): values of classes to extract from segmentation mask
        augmentation (bool): data transfromation pipeline
            (e.g. flip, scale, etc.)
        normalization (bool): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """

    def __init__(
        self,
        data_mode,
        classes=None,
        augmentation=False,
        copy_paste=False,
        normalization=True,
        mode="RGBD",
    ):
        self.ids = os.listdir(SEGDATA_DIR / data_mode)
        self.rgbd_seg = {"rgb": [], "depth": [], "seg": []}
        for obj_id in sorted(self.ids):
            with np.load(SEGDATA_DIR / data_mode / obj_id) as obj:
                print(obj_id)
                for key in self.rgbd_seg.keys():
                    self.rgbd_seg[key].append(obj[key])
        for k in self.rgbd_seg:
            self.rgbd_seg[k] = np.vstack(self.rgbd_seg[k])
        # for key in self.rgbd_seg.keys():
        #     self.rgbd_seg[key] = np.array(self.rgbd_seg[key])

        # rgb: (N, 3, H, W), depth: (N, H, W), seg: (N, H, W)

        # convert str names to class values on masks
        self.class_values = list(range(len(classes)))

        self.augmentation = augmentation
        self.normalization = normalization
        self.data_mode = data_mode
        self.mode = mode
        self.copy_paste = copy_paste

        self.sp_aug = spatial_augmentation(copy_paste=self.copy_paste)
        self.rgb_px_aug = rgb_pixel_augmentation()
        self.d_px_aug = depth_pixel_augmention()

    def __getitem__(self, i):
        """
        self.data: rgb, stereo_depth, clean_depth, seg, bg_seg
        """
        # read data
        ori_image_rgbd = np.concatenate(
            (self.rgbd_seg["rgb"][i], self.rgbd_seg["depth"][i][None]), axis=0
        ).transpose(
            1, 2, 0
        )  # (H, W, 4)
        mask = self.rgbd_seg["seg"][i]  # (H, W)

        # extract certain classes from mask (e.g. cars)
        ori_mask = [(mask == v).astype(np.float32) for v in self.class_values]

        aug_image, aug_mask = None, None

        # apply augmentations
        if self.augmentation and self.data_mode == "train":
            if self.copy_paste is True:
                # paste faucet on door/drawer, vice versa
                is_faucet = (ori_mask[3].sum() + ori_mask[4].sum()) > 0
                paste_is_faucet = is_faucet
                while is_faucet == paste_is_faucet:
                    paste_i = random.randint(0, len(self.rgbd_seg["rgb"]) - 1)
                    paste_masks = [
                        (self.rgbd_seg["seg"][paste_i] == v).astype(np.float32)
                        for v in self.class_values
                    ]  # (H, W, 6)
                    paste_is_faucet = (paste_masks[3].sum() + paste_masks[4].sum()) > 0
                paste_rgbd = np.concatenate(
                    (
                        self.rgbd_seg["rgb"][paste_i],
                        self.rgbd_seg["depth"][paste_i][None],
                    ),
                    axis=0,
                ).transpose(
                    1, 2, 0
                )  # (H, W, 4)
                sample = self.sp_aug(
                    image=ori_image_rgbd,
                    masks=ori_mask,
                    paste_image=paste_rgbd,
                    paste_masks=paste_masks,
                    paste_bboxes=None,
                )  # (H, W, C)
            else:
                sample = self.sp_aug(image=ori_image_rgbd, masks=ori_mask)  # (H, W, C)
            aug_mask = sample["masks"]  # [(H,W)]*6
            if self.mode == "RGBD":
                sample_rgb = self.rgb_px_aug(
                    image=sample["image"][..., :3].astype(np.uint8)
                )  # (H, W, 3)
                sample_d = self.d_px_aug(
                    image=sample["image"][..., 3][..., None]
                )  # (H, W, 1)
                aug_image = np.concatenate(
                    [
                        sample_rgb["image"].transpose(2, 0, 1),
                        sample_d["image"].transpose(2, 0, 1),
                    ],
                    axis=0,
                )  # (4, H, W)
            elif self.mode == "RGD":
                sample_rgb = self.rgb_px_aug(
                    image=sample["image"][..., :3].astype(np.uint8)
                )  # (H, W, 3)
                sample_d = self.d_px_aug(
                    image=sample["image"][..., 3][..., None]
                )  # (H, W, 1)
                aug_image = np.concatenate(
                    [
                        sample_rgb["image"].transpose(2, 0, 1)[:2],
                        sample_d["image"].transpose(2, 0, 1),
                    ],
                    axis=0,
                )  # (3, H, W)
            elif self.mode == "RGB":
                sample_rgb = self.rgb_px_aug(
                    image=sample["image"][..., :3].astype(np.uint8)
                )  # (H, W, 3)
                aug_image = sample_rgb["image"].transpose(2, 0, 1)  # (3, H, W)
            elif self.mode == "D":
                sample_d = self.d_px_aug(
                    image=sample["image"][..., 3][..., None]
                )  # (H, W, 1)
                aug_image = sample_d["image"].transpose(2, 0, 1)  # (1, H, W)
            else:
                raise NotImplementedError
        else:
            if self.mode == "RGBD":
                aug_image, aug_mask = ori_image_rgbd.transpose((2, 0, 1)), ori_mask
            elif self.mode == "RGD":
                aug_image = np.concatenate(
                    (self.rgbd_seg["rgb"][i][:2], self.rgbd_seg["depth"][i][None]),
                    axis=0,
                )  # (3, H, W)
                aug_mask = ori_mask
            elif self.mode == "RGB":
                aug_image, aug_mask = ori_image_rgbd.transpose((2, 0, 1))[:3], ori_mask
            elif self.mode == "D":
                aug_image, aug_mask = ori_image_rgbd.transpose((2, 0, 1))[3:4], ori_mask
            else:
                raise NotImplementedError
        # apply normalization
        if self.normalization:
            if self.mode == "RGBD":
                aug_image[:3] = aug_image[:3] / 255.0
                # aug_image[3] = aug_image[3]/np.max(aug_image[3])
            elif self.mode == "RGD":
                aug_image[:2] = aug_image[:2] / 255.0
            elif self.mode == "RGB":
                aug_image = aug_image / 255.0
            elif self.mode == "D":
                pass
                # aug_image[3] = aug_image[3]/np.max(aug_image[3])
            else:
                raise NotImplementedError

        return ori_image_rgbd, ori_mask, aug_image, aug_mask

    def __len__(self):
        return len(self.rgbd_seg["seg"])


class KeypointDataset(BaseDataset):
    def __init__(
        self,
        data_mode,
        augmentation=False,
        normalization=True,
        mode="RGBD",
    ):
        self.ids = sorted(
            [file[:-4] for file in os.listdir(KPTDATA_DIR / f"{data_mode}/")]
        )

        rgb_data, depth_data, kpts_data = [], [], []
        for obj_id in self.ids:
            rgbd = np.load(KPTDATA_DIR / f"{data_mode}/{obj_id}.npz")
            rgb_data.extend(rgbd["rgb"])
            depth_data.extend(rgbd["depth"][:, None])
            kpts = np.load(
                KPTDATA_DIR / f"{data_mode}_kpts/{obj_id}_keypoints.npy"
            )  # (N, 3, 3, 3)
            kpts_data.extend(kpts)
        self.rgbd_data = np.concatenate((rgb_data, depth_data), axis=1)  # (N, 4, H, W)
        self.kpts_data = np.array(kpts_data)  # (N, J, 3, 3)

        self.augmentation = augmentation
        self.normalization = normalization
        self.mode = mode

    def __getitem__(self, i):
        """
        self.data: rgb, stereo_depth, clean_depth, seg, bg_seg
        """
        # read data
        ori_image_rgbd = self.rgbd_data[i].transpose(1, 2, 0)  # (H, W, 4)
        ori_kpts = self.kpts_data[i, 1]  # (J, 3), uvz
        # ori_kpts = self.kpts_data[i, 0]  # (J, 3), cam_xyz
        ori_kpt_visable = self.kpts_data[i, 2, :, 2]  # (J), uvz visable

        aug_image, aug_kpts, aug_kpt_visable = None, ori_kpts, ori_kpt_visable

        # apply augmentations
        if self.augmentation:
            sample = spatial_augmentation()(image=ori_image_rgbd)  # (H, W, C)
            if self.mode == "RGBD":
                sample_rgb = rgb_pixel_augmentation()(
                    image=sample["image"][..., :3].astype(np.uint8)
                )  # (H, W, 3)
                sample_d = depth_pixel_augmention()(
                    image=sample["image"][..., 3][..., None]
                )  # (H, W, 1)
                aug_image = np.concatenate(
                    [
                        sample_rgb["image"].transpose(2, 0, 1),
                        sample_d["image"].transpose(2, 0, 1),
                    ],
                    axis=0,
                )  # (4, H, W)
            elif self.mode == "RGB":
                sample_rgb = rgb_pixel_augmentation()(
                    image=sample["image"][..., :3].astype(np.uint8)
                )  # (H, W, 3)
                aug_image = sample_rgb["image"].transpose(2, 0, 1)  # (3, H, W)
            elif self.mode == "D":
                sample_d = depth_pixel_augmention()(
                    image=sample["image"][..., 3][..., None]
                )  # (H, W, 1)
                aug_image = sample_d["image"].transpose(2, 0, 1)  # (1, H, W)
            else:
                raise NotImplementedError
        else:
            if self.mode == "RGBD":
                aug_image = ori_image_rgbd.transpose(2, 0, 1)
            elif self.mode == "RGB":
                aug_image = ori_image_rgbd.transpose(2, 0, 1)[:3]
            elif self.mode == "D":
                aug_image = ori_image_rgbd.transpose(2, 0, 1)[3]
            else:
                raise NotImplementedError
        # apply normalization
        if self.normalization:
            ## uvz min, max: (11, 0, 0.185), (242, 143, 0.88)
            kpts_min = np.array([0, 0, 0.18])
            kpts_max = np.array([255, 143, 1.0])
            ## camxyz min, max: (-0.25, -0.19, 0.18), (0.29, 0.30, 0.88)
            # kpts_min = np.array([-0.3, -0.2, 0.18])
            # kpts_max = np.array([0.3, 0.3, 1.0])
            aug_kpts = (aug_kpts - kpts_min) / (kpts_max - kpts_min)
            if self.mode == "RGBD":
                aug_image[:3] = aug_image[:3] / 255.0
                # aug_image[3] = aug_image[3]/np.max(aug_image[3])
            elif self.mode == "RGB":
                aug_image = aug_image / 255.0
            elif self.mode == "D":
                pass
                # aug_image[3] = aug_image[3]/np.max(aug_image[3])
            else:
                raise NotImplementedError

        return ori_image_rgbd, ori_kpts, aug_image, aug_kpts, aug_kpt_visable

    def __len__(self):
        return self.kpts_data.shape[0]


if __name__ == "__main__":
    MODE = "RGBD"  # RGBD, RGB, D
    CLASSES = ["handle", "door", "cabinet", "switchlink", "fixlink", "other"]

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    seg_dataset = SegDataset(
        data_mode="train",
        augmentation=True,
        copy_paste=True,
        classes=CLASSES,
        mode=MODE,
    )

    cnt = 0
    while cnt < 5:
        ind = np.random.choice(range(len(seg_dataset)))
        image_rgbd, mask, aug_image_rgbd, aug_mask = seg_dataset[ind]  # get some sample
        ori_bin_masks = {"o-" + CLASSES[i]: mask[i] for i in range(len(CLASSES))}
        aug_bin_masks = {"a-" + CLASSES[i]: aug_mask[i] for i in range(len(CLASSES))}
        visualize(
            ori_rgb=image_rgbd[:, :, :3] / 255.0,
            ori_depth=image_rgbd[:, :, 3],
            aug_rgb=aug_image_rgbd[:3].transpose(1, 2, 0),
            aug_depth=aug_image_rgbd[3],
            **ori_bin_masks,
            **aug_bin_masks,
        )
        time.sleep(0.5)
        cnt += 1

    # kpts_dataset = KeypointDataset(
    #     data_mode="train",
    #     augmentation=True,
    #     mode=MODE
    # )
    #
    # cnt = 0
    # while cnt < 5:
    #     ind = np.random.choice(range(len(kpts_dataset)))
    #     image_rgbd, kpts, aug_image_rgbd, aug_kpts, aug_kpt_visable = kpts_dataset[ind]  # get some sample
    #     vis_keypoints(aug_image_rgbd[:3].transpose((1, 2, 0)), aug_image_rgbd[3], aug_kpts)
    #     cnt += 1
