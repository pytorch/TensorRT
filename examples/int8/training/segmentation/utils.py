import torch
import torchvision
from dataset import SceneParsingDataset

from torch.utils.data import DataLoader

THRESHOLD = 0.5
EPSILON = 1e-8

def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True
):
    train_ds = SceneParsingDataset(
        image_dir = train_dir,
        mask_dir = train_mask_dir,
        transform = train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True
    )

    val_ds = SceneParsingDataset(
        image_dir = val_dir,
        mask_dir = val_mask_dir,
        transform = val_transform,
        training=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False
    )

    return train_loader, val_loader

def save_checkpoint(state, CHECKPOINT_NAME):
    print("Saving checkpoint")
    torch.save(state, CHECKPOINT_NAME)

def load_checkpoint(checkpoint, model):
    print("Loading model")
    model.load_state_dict(checkpoint["state_dict"])

def pixel_accuracy(pred, y):
    _, preds = torch.max(pred, dim=1)
    valid = (y >= 0).long()
    acc_sum = torch.sum(valid * (preds == y).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

def check_accuracy(loader, model, device="cuda"):
    correct = 0
    total_count = 0    # TODO: Explore DICE score
    dice_score = []
    model.eval()
    acc_sum = []
    iou_score = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().to(device)

            pred = model(x)
            _, preds = torch.max(pred, dim=1)

            match = (preds == y).sum()
            npixels = torch.numel(y) + torch.numel(preds >= 0)

            iou = match / npixels
            iou_score.append(iou)

            dice_score.append(2. * (match/npixels) + EPSILON)
            '''
            valid = (y >= 0).long()
            correct += (preds == y).sum()
            total_count += torch.numel(valid)
            
            dice_score += (2 * (preds == y).sum() / (preds * y).sum() + EPSILON)
            '''
    # print(f"Result {correct}/{total_count} with accuracy {correct/total_count * 100:.2f}")
    # print(f"Dice Score: {dice_score/len(loader)}")
    print(f"Mean IoU: {sum(iou_score)/len(iou_score):.2f}")
    print(f"Dice score: {sum(dice_score)/len(dice_score):.2f}")

