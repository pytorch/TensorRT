import argparse
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.tensorboard import SummaryWriter

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib
from tqdm import tqdm

from vgg16 import vgg16

PARSER = argparse.ArgumentParser(
    description="VGG16 example to use with Torch-TensorRT PTQ"
)
PARSER.add_argument(
    "--epochs", default=100, type=int, help="Number of total epochs to train"
)
PARSER.add_argument(
    "--enable_qat",
    action="store_true",
    help="Enable quantization aware training. This is recommended to perform on a pre-trained model.",
)
PARSER.add_argument(
    "--batch-size", default=128, type=int, help="Batch size to use when training"
)
PARSER.add_argument("--lr", default=0.1, type=float, help="Initial learning rate")
PARSER.add_argument("--drop-ratio", default=0.0, type=float, help="Dropout ratio")
PARSER.add_argument("--momentum", default=0.9, type=float, help="Momentum")
PARSER.add_argument("--weight-decay", default=5e-4, type=float, help="Weight decay")
PARSER.add_argument(
    "--ckpt-dir",
    default="/tmp/vgg16_ckpts",
    type=str,
    help="Path to save checkpoints (saved every 10 epochs)",
)
PARSER.add_argument(
    "--start-from",
    default=0,
    type=int,
    help="Epoch to resume from (requires a checkpoin in the providied checkpoi",
)
PARSER.add_argument("--seed", type=int, help="Seed value for rng")
PARSER.add_argument(
    "--tensorboard",
    type=str,
    default="/tmp/vgg16_logs",
    help="Location for tensorboard info",
)

args = PARSER.parse_args()
for arg in vars(args):
    print(" {} {}".format(arg, getattr(args, arg)))
state = {k: v for k, v in args._get_kwargs()}

if args.seed is None:
    args.seed = random.randint(1, 10000)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
print("RNG seed used: ", args.seed)

now = datetime.now()

timestamp = datetime.timestamp(now)

writer = SummaryWriter(args.tensorboard + "/test_" + str(timestamp))
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(f"{name:40}: {module}")
    model.cuda()


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def calibrate_model(
    model,
    model_name,
    data_loader,
    num_calib_batch,
    calibrator,
    hist_percentile,
    out_dir,
):
    """
    Feed data to the network and calibrate.
    Arguments:
        model: classification model
        model_name: name to use when creating state files
        data_loader: calibration data set
        num_calib_batch: amount of calibration passes to perform
        calibrator: type of calibration to use (max/histogram)
        hist_percentile: percentiles to be used for historgram calibration
        out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                f"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth",
            )
            torch.save(model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(f"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    f"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth",
                )
                torch.save(model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(f"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(
                    out_dir,
                    f"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth",
                )
                torch.save(model.state_dict(), calib_output)


def main():

    global state
    global classes
    global writer
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    training_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    testing_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )

    testing_dataloader = torch.utils.data.DataLoader(
        testing_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    num_classes = len(classes)

    quant_modules.initialize()

    model = vgg16(num_classes=num_classes, init_weights=False)
    model = model.cuda()

    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.start_from != 0:
        ckpt_file = args.ckpt_dir + "/ckpt_epoch" + str(args.start_from) + ".pth"
        print("Loading from checkpoint {}".format(ckpt_file))
        assert os.path.isfile(ckpt_file)
        ckpt = torch.load(ckpt_file)
        modified_state_dict = {}
        for key, val in ckpt["model_state_dict"].items():
            # Remove 'module.' from the key names
            if key.startswith("module"):
                modified_state_dict[key[7:]] = val
            else:
                modified_state_dict[key] = val

        model.load_state_dict(modified_state_dict)
        opt.load_state_dict(ckpt["opt_state_dict"])
        state = ckpt["state"]

    data = iter(training_dataloader)
    images, _ = next(data)

    writer.add_graph(model, images.cuda())
    writer.close()

    # ## Calibrate the model
    with torch.no_grad():
        calibrate_model(
            model=model,
            model_name="vgg16",
            data_loader=training_dataloader,
            num_calib_batch=32,
            calibrator="max",
            hist_percentile=[99.9, 99.99, 99.999, 99.9999],
            out_dir="./",
        )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(args.start_from, args.epochs):
        adjust_lr(opt, epoch)
        writer.add_scalar("Learning Rate", state["lr"], epoch)
        writer.close()
        print("Epoch: [%5d / %5d] LR: %f" % (epoch + 1, args.epochs, state["lr"]))

        train(model, training_dataloader, crit, opt, epoch)
        test_loss, test_acc = test(model, testing_dataloader, crit, epoch)

        print("Test Loss: {:.5f} Test Acc: {:.2f}%".format(test_loss, 100 * test_acc))

        if epoch % 10 == 9 or epoch == args.epochs - 1:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "acc": test_acc,
                    "opt_state_dict": opt.state_dict(),
                    "state": state,
                },
                ckpt_dir=args.ckpt_dir,
            )


def train(model, dataloader, crit, opt, epoch):
    global writer
    model.train()
    running_loss = 0.0
    for batch, (data, labels) in enumerate(dataloader):
        data, labels = data.cuda(), labels.cuda(non_blocking=True)
        opt.zero_grad()
        out = model(data)
        loss = crit(out, labels)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if batch % 50 == 49:
            writer.add_scalar(
                "Training Loss", running_loss / 100, epoch * len(dataloader) + batch
            )
            writer.close()
            print(
                "Batch: [%5d | %5d] loss: %.3f"
                % (batch + 1, len(dataloader), running_loss / 100)
            )
            running_loss = 0.0


def test(model, dataloader, crit, epoch):
    global writer
    global classes
    total = 0
    correct = 0
    loss = 0.0
    class_probs = []
    class_preds = []
    model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = model(data)
            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            class_probs.append([F.softmax(i, dim=0) for i in out])
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    writer.add_scalar("Testing Loss", loss / total, epoch)
    writer.close()

    writer.add_scalar("Testing Accuracy", correct / total * 100, epoch)
    writer.close()

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds, epoch)
    print(loss, total, correct, total)
    return loss / total, correct / total


def save_checkpoint(state, ckpt_dir="checkpoint"):
    print("Checkpoint {} saved".format(state["epoch"]))
    filename = "ckpt_epoch" + str(state["epoch"]) + ".pth"
    filepath = os.path.join(ckpt_dir, filename)
    torch.save(state, filepath)


def adjust_lr(optimizer, epoch):
    global state
    new_lr = state["lr"] * (0.5 ** (epoch // 40)) if state["lr"] > 1e-7 else state["lr"]
    if new_lr != state["lr"]:
        state["lr"] = new_lr
        print("Updating learning rate: {}".format(state["lr"]))
        for param_group in optimizer.param_groups:
            param_group["lr"] = state["lr"]


def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    global classes
    """
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    """
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(
        classes[class_index],
        tensorboard_preds,
        tensorboard_probs,
        global_step=global_step,
    )
    writer.close()


if __name__ == "__main__":
    main()
