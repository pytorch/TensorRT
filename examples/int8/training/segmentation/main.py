import os
import random
import torch

from model import VGG16Unet
from dataset import NUM_CLASSES
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim

from PIL import Image
import argparse
from tqdm import tqdm

from utils import (
    get_loaders,
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    compute_dice_score,
    get_loss_fn,
)

IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_WORKERS = 2
PIN_MEMORY = True
TRAIN_IMG_DIR = "ADEChallengeData2016/images/training"
TRAIN_MASK_DIR = "ADEChallengeData2016/annotations/training"

VAL_IMG_DIR = "ADEChallengeData2016/images/validation"
VAL_MASK_DIR = "ADEChallengeData2016/annotations/validation"

CKPT_DIR = "model_ckpt"
CHECKPOINT_PREFIX = "vgg16_unet"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = random.randint(1, 100000)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def train(loader, model, optimizer, loss_fn, scaler):
    pr_loop = tqdm(loader)

    train_loss = 0.0

    for data, target in pr_loop:
        data = data.to(device=DEVICE)

        target = target.to(device=DEVICE).long()

        preds = model(data)
        loss = loss_fn(preds, target)

        # dice = compute_dice_score(preds, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * data.size(0)
        pr_loop.set_postfix(ordered_dict={"loss": loss.item()})
        # , "dice_score": dice.item()})

    return train_loss  # , dice


def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--data", type=str, default="", help="Path to dataset root dir")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--interval", type=int, default=5, help="Epoch intervals to save checkpoint"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument(
        "--start-from", type=int, default=0, help="Load the checkpoint epoch"
    )
    parser.add_argument(
        "--export",
        type=str,
        default="segmentation_model.jit.pt",
        help="Export as a Torch Script",
    )
    parser.add_argument(
        "--load-model", type=bool, default=True, help="Load a checkpoint"
    )

    args = parser.parse_args()

    # Transform for training set
    train_transform = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Transform for validation set
    val_transform = A.Compose(
        [
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = VGG16Unet().to(device=DEVICE)

    # Multi-class semantic segmentation
    loss_fn = get_loss_fn()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Pick dataset root directory according as per user input if required
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR if args.data == "" else os.path.join(args.data, TRAIN_IMG_DIR),
        TRAIN_MASK_DIR if args.data == "" else os.path.join(args.data, TRAIN_MASK_DIR),
        VAL_IMG_DIR if args.data == "" else os.path.join(args.data, VAL_IMG_DIR),
        VAL_MASK_DIR if args.data == "" else os.path.join(args.data, VAL_MASK_DIR),
        args.batch,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    check_accuracy(val_loader, model, device=DEVICE)

    epochs = 0
    ckpt_file = None
    if args.load_model == True and os.path.exists(CKPT_DIR):
        for ckpt in os.listdir(CKPT_DIR):
            if ckpt.startswith(CHECKPOINT_PREFIX):
                names = ckpt.split("_")
                if names:
                    names = names[-1].split(".pth.tar")[0]

                epochs = max(epochs, int(names))
                ckpt_file = ckpt

        if ckpt_file is not None and os.path.exists(ckpt_file):
            load_checkpoint(torch.load(ckpt_file), model)

    scaler = torch.cuda.amp.GradScaler()
    checkpoint = dict()
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    for epoch in range(epochs, args.epochs):
        print(f"Epoch: {epoch}/{args.epochs} ...")
        train_loss = train(train_loader, model, optimizer, loss_fn, scaler)

        if epoch % args.interval == 0:
            val_loss, dice_score = check_accuracy(val_loader, model, device=DEVICE)
            print(
                f"Train loss: {train_loss/len(train_loader.dataset):.4f} Val loss: {val_loss:.4f} Dice Score: {dice_score:.4f}"
            )
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            ckpt_filename = CHECKPOINT_PREFIX + "_epoch_" + str(epoch) + ".pth.tar"
            save_checkpoint(checkpoint, os.path.join(CKPT_DIR, ckpt_filename))

    ckpt_filename = CHECKPOINT_PREFIX + "_epoch_" + str(args.epochs) + ".pth.tar"
    save_checkpoint(checkpoint, os.path.join(CKPT_DIR, ckpt_filename))

    model.eval()
    # Check accuracy
    val_loss, dice = check_accuracy(val_loader, model, device=DEVICE)
    print(f"Evaluation loss: {val_loss:.4f} Dice Score: {dice:.4f}")

    if args.export:
        mod = torch.jit.script(model)
        torch.jit.save(mod, "vgg16unet.jit.pt")


if __name__ == "__main__":
    main()
