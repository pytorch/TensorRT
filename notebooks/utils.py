import time
import numpy as np
import wget
import tarfile
import torch.backends.cudnn as cudnn
import os
import shutil

cudnn.benchmark = True

# Helper function to benchmark the model
def benchmark(model, input_shape=(1024, 1, 32, 32), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            output = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%100==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output shape:", output.shape)
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))


def download_data(DATA_DIR):
    if os.path.exists(DATA_DIR):
        if not os.path.exists(os.path.join(DATA_DIR, 'imagenette2-320')):
            url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
            wget.download(url)
            # open file
            file = tarfile.open('imagenette2-320.tgz')
            # extracting file
            file.extractall(DATA_DIR)
            file.close()
    else:
        print("This directory doesn't exist. Create the directory and run again")


def save_checkpoint(state, ckpt_path="checkpoint.pth"):
    torch.save(state, ckpt_path)
    print("Checkpoint saved")


# +
#Define functions for training, evalution, saving checkpoint and train parameter setting function
def train(model, dataloader, crit, opt, epoch):
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
        if batch % 100 == 99:
            print("Batch: [%5d | %5d] loss: %.3f" % (batch + 1, len(dataloader), running_loss / 100))
            running_loss = 0.0
        
def evaluate(model, dataloader, crit, epoch):
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

    evaluate_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    evaluate_preds = torch.cat(class_preds)

    return loss / total, correct / total

def save_checkpoint(state, ckpt_path="checkpoint.pth"):
    torch.save(state, ckpt_path)
    print("Checkpoint saved")
    
#This function allows you to set the all the parameters to not have gradients, 
#allowing you to freeze the model and not undergo training during the train step. 
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
