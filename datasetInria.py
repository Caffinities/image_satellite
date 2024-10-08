import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
import kornia.augmentation as K
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import transformers
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


class InriaDataset(Dataset):
    def __init__(self, data_dir="InriaImage/AerialImageDataset/train",
                 img_dir="images", 
                 mask_dir="gt", 
                 img_size = 512, # input image size (cropped from original 5k*5k image)
                 resize = False,
                 preprocessor : transformers.image_processing_utils.BaseImageProcessor=
                 Mask2FormerImageProcessor(ignore_index=-100,
                                         reduce_labels=False,
                                         do_resize=False,
                                         do_rescale=False,
                                         do_normalize=False,)):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.name_list = os.listdir(os.path.join(data_dir, img_dir))
        self.preprocessor = preprocessor
        self.resize = resize
        self.transforms = K.AugmentationSequential(
                # K.RandomRotation((-360, 360), p=0.5),
                K.RandomCrop((img_size, img_size), p=1, keepdim=False),
                K.RandomVerticalFlip(),
                K.RandomHorizontalFlip(),
                data_keys=['input', 'mask'],
            )
        scale_factor = 0.3 / 0.5 # 0.3m/pixel if resize==False, 0.5m/pixel if resize==True.
        new_width = int(5000 * scale_factor) # for an original 5k*5k image
        new_height = int(5000 * scale_factor)
        if self.resize:
            self.img_transforms = T.Compose([
                T.Resize((new_height, new_width)),
                T.ToTensor()
            ])
        else:
            self.img_transforms = T.Compose([
                T.ToTensor()
            ])

    def load_img_and_mask(self, index):
        name = self.name_list[index]
        img_path = os.path.join(self.data_dir, self.img_dir, name)
        mask_path = os.path.join(self.data_dir, self.mask_dir, name)
        img = self.img_transforms(Image.open(img_path))
        mask = self.img_transforms(Image.open(mask_path))
        img, mask = self.transforms(img, mask)
        mask = mask.to(torch.int)
        return img, mask

    def __getitem__(self, index):
        img, mask = self.load_img_and_mask(index)
        batch = self.preprocessor(img.squeeze(),
                                  segmentation_maps=mask.squeeze(),
                                  return_tensors='pt')
        batch['pixel_values'] = batch['pixel_values'].squeeze()
        batch["original_segmentations"] = mask.squeeze()
        if len(batch['class_labels'][0])<2:
            batch['mask_labels'] = F.pad(batch['mask_labels'][0], (0, 0, 0, 0, 0, 1))
        batch['mask_labels'] = torch.tensor(np.array(batch['mask_labels'])).squeeze()
        batch['class_labels'] = torch.tensor([0, 1])
        return batch
    
    def __len__(self):
        return len(self.name_list)