from glob import glob

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time
from IPython.display import clear_output
import tqdm
from tqdm import tqdm
import numpy as np
from pathlib import Path

### check: https://www.kaggle.com/mlagunas/naive-unet-with-pytorch-tensorboard-logging

def to_np(x):
    """
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L20
    :param x:
    :return:
    """
    return x.data.cpu().numpy()

def dic_name(outpath, LEN=9):
    files = sorted(glob(outpath+"*/"))
    if len(files) == 0:
        name=str(10**(LEN+1)+1)[1:]
    else:
        nr  = len(files)+1
        name= str(10**(LEN+1)+nr)[1:]

    return outpath+name+"/"


# define a class to log values during training
class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BCELoss2d(nn.Module):
    """
    Code taken from:
    https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208
    """

    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


def train(model, train_dl, valid_dl, device, optimizer, acc_fn, epochs=1,writer=None,wlogpath="weights/"):
    start  = time.time()
    loss_fn= nn.CrossEntropyLoss()##nn.BCELoss()##BCELoss2d().to(device)
    losses = AverageMeter()

    train_loss, valid_loss = [], []
    acc_val = []

    best_acc = 0.0


    wlog = dic_name(wlogpath)
    Path(wlog).mkdir(parents=True, exist_ok=True)


    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0



            # iterate over data
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (x,y) in pbar:
                x = x.to(device)
                y = y.to(device)


                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    losses.update(loss.data, x.size(0))

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()

                    if writer != None:
                        #writer.add_graph(model, outputs)
                        # log loss values every iteration
                        writer.add_scalar('data/(train)loss_val', losses.val, i + 1)
                        writer.add_scalar('data/(train)loss_avg', losses.avg, i + 1)
                        # log the layers and layers gradient histogram and distributions
                        for tag, value in model.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('model/(train)' + tag, to_np(value), i + 1)
                            writer.add_histogram('model/(train)' + tag + '/grad', to_np(value.grad), i + 1)
                        # log the outputs given by the model (The segmentation)
                        #writer.add_image('model/(train)output', make_grid(outputs.data), i + 1)

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())
                        losses.update(loss.data, x.size(0))
                        if writer != None:
                            writer.add_scalar('data/(test)loss_val', losses.val, i + 1)
                            writer.add_scalar('data/(test)loss_avg', losses.avg, i + 1)
                            for tag, value in model.named_parameters():
                                tag = tag.replace('.', '/')
                                writer.add_histogram('model/(test)' + tag, to_np(value), i + 1)
                                writer.add_histogram('model/(test)' + tag + '/grad', to_np(value.grad), i + 1)
                            # log the outputs given by the model (The segmentation)
                            #writer.add_image('model/(test)output', make_grid(outputs.data), i + 1)
                #writer.add_scalar("Loss/valid", loss, epoch)

                acc = acc_fn(outputs, y, device)

                running_acc  += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size

                if i % 100 == 0:
                    # clear_output(wait=True)
                    tqdm.write('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(i, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    #print(torch.cuda.memory_summary())

                    ### model.module.state_dict() for multiGPU
                    torch.save(model.state_dict(), wlog+str(10000000+epoch+1)[1:]+"_"+str(10+i)[1:]+".pt")

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(float(epoch_loss)) if phase=='train' else valid_loss.append(float(epoch_loss))
            acc_val.append(float(epoch_acc))


        torch.save(model.state_dict(), wlog+str(10000000+epoch+1)[1:]+".pt")

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss, acc_val, writer
