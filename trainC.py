from tqdm.auto import tqdm
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from accelerate import Accelerator
from datasetInria import InriaDataset
import evaluate
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# torch.manual_seed(1023)
metric = evaluate.load("mean_iou")

scale = 'small' # scale of model: choice between tiny, small, base, large. 50M, 100M, 200M.
img_size = 512 # input image size (cropped from original 5k*5k image)
epochs = 200
resize = False # whether change the resolution of image or not, 0.3m/pixel if False, 0.5m/pixel if True.
patience = 50
train_ratio = 0.9
learning_rate = 1e-4
weight_decay = 0.01

accelerator = Accelerator()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = Mask2FormerForUniversalSegmentation.from_pretrained(f"facebook/mask2former-swin-{scale}-ade-semantic",)

# model = torch.nn.DataParallel(model)
# model.load_state_dict(torch.load('Mask2_A_base_120.pt', map_location="cpu"))
# model = torch.load("Mask2Former_tiny_25.pt", map_location='cpu')

model = model.to(device)
dataset = InriaDataset(img_size=img_size, resize=resize)
train_dataset, test_dataset = random_split(dataset, [train_ratio, 1.0-train_ratio])
train_dataloader = DataLoader(train_dataset, batch_size=4)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# batch_size for tiny: 6, 8GiB, for small: 4, 6.1GiB.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,)# weight_decay=0.05,
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/10.0)
model, optimizer, train_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, scheduler
)

# initialization
epoch_taken = 0
best_MIoU = 0.0
epochs_trained = 0
name = f"Mask2_B_resize{resize}_crop{img_size}_{scale}_{epochs}.pt"

for epoch in tqdm(range(epochs)):
    # if epoch%10 == 0:
    #     model.eval()
    #     for i, batch in enumerate(test_dataloader):
    #         with torch.no_grad():
    #             outputs = model(pixel_values=batch["pixel_values"].to(torch.float16).to(device),)
    #         target_sizes = [[dataset.img_size, dataset.img_size] for i in batch["pixel_values"]]
    #         predicted_segmentation_maps = dataset.preprocessor.post_process_semantic_segmentation(outputs,
    #                                                                                         target_sizes = target_sizes)
    #         metric.add_batch(references=batch["original_segmentations"], predictions=predicted_segmentation_maps)

    #     test_MIoU = metric.compute(num_labels = 2, ignore_index = -100)['mean_iou']

    #     print(f"\nTest MIoU: ", test_MIoU)
    #     if test_MIoU>best_MIoU:
    #         best_MIoU = test_MIoU
    #         epoch_taken = 0
    #         dtype = model.module.dtype
    #         torch.save(model.module.to(torch.float), name)
    #         model.module.to(dtype)
    #         # if bug exists, use copy.deepcopy(model)

    #     else:
    #         epoch_taken += 1
    #     if epoch_taken >= patience:
    #         break
    
    model.train()
    for i, batch in enumerate((train_dataloader)):
        optimizer.zero_grad()
        pixel_values = batch["pixel_values"].to(torch.float16).to(device)
        # the dtype changes is for a bug in mismatch of dtype between model and batch["mask_labels"]
        class_labels = batch["class_labels"]
        mask_labels = [label.to(device)[:len(class_labels[i])] for i, label in enumerate(batch["mask_labels"])]
        outputs = model(pixel_values=pixel_values,
                        mask_labels=mask_labels,
                        class_labels=class_labels,
                        )
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()

    scheduler.step()
    epochs_trained += 1

print(f"\n {name}")
print(f"epochs trained: {epochs_trained}")
torch.save(model.module.to(torch.float), name)