import torch
from torch.utils.data import Dataset
from einops import rearrange
import torchvision.transforms as T
import kornia.augmentation as K
from torchgeo.datasets import DFC2022
import numpy as np
import transformers
from torch.utils.data import DataLoader
from torchgeo.datamodules.utils import dataset_split
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

class ImageSegementationDataset(Dataset):
    """Image Segmentation Dataset"""
    def __init__(self, split = 'train'):
        self.patch_size = 512
        self.split = split
        
        def preprocess(sample):
            # RGB uint8 -> float32
            sample['image'][:3] /= 255.0
            # discard DEM channel
            sample['image'] = sample['image'][:3]

            if "mask" in sample:
                # ignore the clouds and shadows class
                sample['mask'][sample['mask'] == 15] = 0
                sample['mask'] = rearrange(sample['mask'], 'h w -> () h w')
            return sample
        
        def crop(sample):
            if "mask" in sample:
                sample['mask'] = sample['mask'].to(torch.float)
                sample['image'], sample['mask'] = K.AugmentationSequential(
                    K.RandomCrop((self.patch_size, self.patch_size), p=1, keepdim=False),
                    data_keys=['input', 'mask'],
                )(sample['image'], sample['mask'])
                sample["mask"] = sample['mask'].to(torch.long)
                sample['image'] = rearrange(sample['image'], '() c h w -> c h w')
                sample['mask'] = rearrange(sample['mask'], '() c h w -> c h w')
            else:
                sample['image'] = K.AugmentationSequential(
                    K.RandomCrop((self.patch_size, self.patch_size), p=1, keepdim=False),
                    data_keys=['input'],
                )(sample['image'])
                sample['image'] = rearrange(sample['image'], '() c h w -> c h w')
            return sample

        transforms = T.Compose([preprocess, crop,])
        self.dataset = DFC2022(root="./", split=self.split, transforms=transforms)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        batch = self.dataset.__getitem__(idx)
        count = 0

        # Make sure the mask has at least one non-zero value
        if self.split != 'train':
            return batch['image'], batch['image'], batch['image']
        
        while((torch.unique(batch['mask']) == 0).all() and count < 5):
            batch = self.dataset.__getitem__(idx)
            count += 1
        if(count == 5):
            if(idx == len(self.dataset) - 1):
                return self.__getitem__(0)
            else:
                return self.__getitem__(idx+1)
        return batch['image'], batch['mask'].squeeze(), np.array(batch['mask'].squeeze())
    
class DFC2022Dataloader():
    def __init__(self,
                 preprocessor : transformers.image_processing_utils.BaseImageProcessor
                   = Mask2FormerImageProcessor(ignore_index=0,
                                         reduce_labels=False,
                                         do_resize=False,
                                         do_rescale=False,
                                         do_normalize=False,),
                split = 'train'):
        self.preprocessor = preprocessor
        self.dataset = ImageSegementationDataset(split)
        
    def collate_fn(self, batch):
        inputs = list(zip(*batch))
        images = inputs[0]
        seg_maps = inputs[1]
        batch = self.preprocessor(images,
                            segmentation_maps=seg_maps,
                            return_tensors="pt")
        batch["original_segmentations"] = inputs[2]
        return batch
    
    def get_dataset(self, test_percentage : float = 0.1):
        train_dataset, test_dataset, _ = dataset_split(self.dataset, val_pct=test_percentage, test_pct=0, )
        return train_dataset, test_dataset
    
    def get_dataloader(self, test_percentage : float = 0.1,
                       batch_size : int = 2) -> tuple[DataLoader, DataLoader]:
        train_dataset, test_dataset, _ = dataset_split(self.dataset, val_pct=test_percentage, test_pct=0, )
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=True, 
                                      collate_fn=self.collate_fn)
        test_dataloader = DataLoader(test_dataset,
                                      batch_size=batch_size, 
                                      shuffle=False, 
                                      collate_fn=self.collate_fn)

        return train_dataloader, test_dataloader
