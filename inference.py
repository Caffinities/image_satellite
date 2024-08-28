import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from transformers import Mask2FormerImageProcessor
import cv2

def split(image : torch.Tensor,
          block_size : int,
          size_used : int):
    '''
    Split an image into blocks for inference.
    size_used is the stride of the blocks,
    letting the blocks overlap and avoid the boundary effect.
    '''
    width, height = image.shape[2], image.shape[1]
    rows = (height + size_used - 1) // size_used
    cols = (width + size_used - 1) // size_used
    image_tensor = image[:3]
    padded_image_tensor = F.pad(image_tensor,
                                (0, size_used * cols - width, 0, size_used * rows - height))
    diff = (block_size-size_used)//2
    padded_image_tensor = F.pad(padded_image_tensor,
                                (diff, diff, diff, diff))
    blocks = F.unfold(padded_image_tensor,
                      kernel_size=(block_size, block_size),
                      stride=(size_used, size_used),)
    blocks = rearrange(blocks, "(c h w) num -> num c h w",
                       c=image_tensor.shape[0],
                       h=block_size).contiguous()

    return blocks

def merge_blocks(blocks : torch.Tensor,
                 original_width : int,
                 original_height : int,
                 block_size : int = 300):

    rows = (original_height + block_size - 1) // block_size
    cols = (original_width + block_size - 1) // block_size
    blocks = blocks.to(torch.float)
    blocks = rearrange(blocks, "num h w -> (h w) num")
    output_size = (block_size * rows, block_size * cols)
    merged_image = F.fold(input=blocks,
                          output_size=output_size,
                          kernel_size=block_size,
                          stride=block_size)
    # crop to original size
    final_image = merged_image[:, :original_height, :original_width]
    
    return final_image.squeeze()

class InferencePipeline():
    def __init__(self,
                 model_path : str="weights/Mask2_B_resizeFalse_crop512_tiny_150.pt",
                 preprocessor = Mask2FormerImageProcessor(ignore_index=-100,
                                                          reduce_labels=False,
                                                          do_resize=False,
                                                          do_rescale=False,
                                                          do_normalize=False)):
        
        self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.preprocessor = preprocessor

    def get_image(self,
                  image_path : str,
                  resolution : float = 0.3) -> np.array:
        image = Image.open(image_path)
        transform = T.Compose([T.Resize((int(image.size[1]*resolution/0.3),
                                         int(image.size[0]*resolution/0.3))),
                               T.ToTensor()]) # assume the model is trained with 0.3m/pixel
        image = transform(image)
        
        return image.permute(1, 2, 0).numpy()

    def get_seg(self,
              blocks : torch.Tensor,
              image_size : int = 512) -> torch.Tensor:
        '''
        For a batch of blocks, return the segmentation mask.
        '''
        block_loader = torch.utils.data.DataLoader(blocks, batch_size=1)
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for i, sub_block in tqdm(enumerate(block_loader), total=len(block_loader)):
                output = self.model(pixel_values=sub_block.to(self.device))

                mask_pred = self.preprocessor.post_process_semantic_segmentation(output,
                                                                target_sizes=[[image_size, image_size]])
                outputs.append(mask_pred[0].cpu())
        results = torch.stack(outputs)

        return results
    
    def segment(self,
                image : torch.Tensor,
                image_size : int = 512,
                size_used : int = 300) -> torch.Tensor:
        '''
        For an entire image, return the overall segmentation mask.

        Parameters:
        ----------
        image_size: int
            Input image size to the model, by default 512.
        size_used: int
            The stride of the blocks, to avoid the boundary effect.
        '''
        blocks = split(image, image_size, size_used)
        out_blocks = self.get_seg(blocks, image_size)
        diff = (image_size - size_used)//2
        out_blocks = out_blocks[:, diff:size_used+diff, diff:size_used+diff].contiguous()
        segmentation = merge_blocks(out_blocks, image.shape[2], image.shape[1], size_used)

        return segmentation
    
    def compare(self,
                image0_path : str,
                image1_path : str,
                resolution0 : float = 0.3,
                resolution1 : float = 0.3,
                img_size : int = 512,
                size_used : int = 300) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Compare two images and return the segmentation masks.
        `resolution0` is necessary to ensure the two images have the same resolution.

        Parameters
        ----------
        resolution0: float
            By default 0.3m/pixel

        '''
        image0 = Image.open(image0_path)
        image1 = Image.open(image1_path)
        transform0 = T.Compose([T.Resize((int(image0.size[1]*resolution0/0.3), int(image0.size[0]*resolution0/0.3))),
                                T.ToTensor()])
        
        transform1 = transform0 # ensure they have the same resolution
        image0 = transform0(image0)
        image1 = transform1(image1)
        seg0 = self.segment(image0, img_size, size_used)
        seg1 = self.segment(image1, img_size, size_used)

        return seg0, seg1
    
    def process(self,
                image0 : np.array,
                image1 : np.array,
                kernel : np.array = np.eye(11),
                kernel2 : np.array = np.fliplr(np.eye(11)),
                kernel_rec : np.array = np.ones((5, 5))):
        '''
        Porcess using different kernels, not used in the final version.
        '''
        filtered = cv2.filter2D(image1-image0, -1, kernel/np.sum(kernel))
        filtered = np.where(filtered > 0.8, 1., np.where(filtered < -0.8, -1., 0.))

        filtered = cv2.filter2D(filtered, -1, kernel2/np.sum(kernel2))
        filtered = np.where(filtered > 0.8, 1., np.where(filtered < -0.8, -1., 0.))

        filtered = cv2.filter2D(filtered, -1, kernel_rec)
        filtered = np.where(filtered >= 1., 1., np.where(filtered <= -1., -1., 0.))

        return filtered