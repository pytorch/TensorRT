import torch
from dataset import NUM_CLASSES, SceneParsingDataset

from torch.utils.data import DataLoader

THRESHOLD = 0.5
EPSILON = 1e-7


def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True,
):
    train_ds = SceneParsingDataset(
        image_dir=train_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SceneParsingDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
        training=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def save_checkpoint(state, CHECKPOINT_NAME):
    print("Saving checkpoint")
    torch.save(state, CHECKPOINT_NAME)


def load_checkpoint(checkpoint, model):
    print("Loading model")
    model.load_state_dict(checkpoint["state_dict"])


def compute_dice_score(pred, label):
    pred = torch.nn.functional.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    pred = (pred > THRESHOLD).float()

    correct = torch.sum(pred == label)
    union = torch.sum(pred >= 0) + torch.sum(label)

    score = (2.0 * correct) / (union + EPSILON)
    return score


def get_loss_fn():
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=150)
    return loss_fn


def check_accuracy(loader, model, device="cuda"):
    dice_score = 0.0

    correct = 0
    pixels = 0
    val_loss = 0.0

    loss_fn = get_loss_fn()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            val_loss += loss_fn(pred, y.long())

            pred = torch.nn.functional.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = (pred > THRESHOLD).float()

            correct += (pred == y).sum()
            pixels += torch.numel(pred)

            dice_score += 2 * (pred * y).sum() / ((pred + y).sum() + EPSILON)

    return val_loss / len(loader.dataset), dice_score / len(loader.dataset)
