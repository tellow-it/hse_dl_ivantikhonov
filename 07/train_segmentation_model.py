import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 1. Prepare Dataset
class VOCSegmentationDataset(torchvision.datasets.VOCSegmentation):
    def __init__(self, root, image_set='train', transforms=None):
        super().__init__(root, year='2012', image_set=image_set, download=True)
        self.transforms = transforms

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.transforms:
            img = self.transforms(img)
            target = T.ToTensor()(target) * 255  # Convert to [0, 1] and rescale
            target = target.long().squeeze(0)
        return img, target

# Data augmentations
train_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomResizedCrop(size=(256, 256), scale=(0.5, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
root_dir = '.'
train_dataset = VOCSegmentationDataset(root=root_dir, image_set='train', transforms=train_transforms)
val_dataset = VOCSegmentationDataset(root=root_dir, image_set='val', transforms=val_transforms)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

# 2. Define Model
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, 21, kernel_size=1)  # PASCAL VOC has 21 classes
model = model.to("cuda")

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# 3. Training and Validation
num_epochs = 5
train_losses, val_losses, miou_scores = [], [], []

def mean_iou(pred, target, num_classes=21):
    iou = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            iou.append(float('nan'))
        else:
            iou.append(intersection / union)
    return np.nanmean(iou)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to("cuda"), masks.to("cuda")
        optimizer.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0
    miou = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to("cuda"), masks.to("cuda")
            outputs = model(imgs)['out']
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            miou += mean_iou(outputs, masks)

    val_losses.append(val_loss / len(val_loader))
    miou_scores.append(miou / len(val_loader))

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, mIoU: {miou_scores[-1]:.4f}")

# 4. Visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(miou_scores, label='mIoU')
plt.legend()
plt.title('Mean IoU')

plt.show()

# Display some validation results
def visualize_predictions(model, loader):
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs.to("cuda")
    with torch.no_grad():
        outputs = model(imgs)['out']
    preds = torch.argmax(outputs, dim=1).cpu()

    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    for i in range(4):
        axes[i, 0].imshow(imgs[i].cpu().permute(1, 2, 0))
        axes[i, 1].imshow(masks[i].cpu(), cmap='gray')
        axes[i, 2].imshow(preds[i], cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 2].set_title('Prediction')
    plt.tight_layout()
    plt.show()

visualize_predictions(model, val_loader)
