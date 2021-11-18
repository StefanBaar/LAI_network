import numpy as np
import torch

from glob import glob
from pathlib import Path

import cv2
from tqdm import tqdm

from joblib import Parallel, delayed

def mCPU(func, var, n_jobs=30,verbose=10):
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(i) for i in var)

def torch_inference(IMAGE,model,GPU=3):
    """Input: Image with shape: (3,n,m)"""
    model = model.cuda(GPU)
    mask = model(IMAGE.unsqueeze(0).cuda(GPU))
    p    = torch.functional.F.softmax(mask[0], 0)
    return p.argmax(0).cpu()

def inference(IMAGE,model,GPU=3):
    IMAGE = IMAGE.transpose((2,0, 1))
    IMAGE = (IMAGE / np.iinfo(IMAGE.dtype).max)
    IMAGE = torch.tensor(IMAGE, dtype=torch.float32)
    return np.asarray(torch_inference(IMAGE,model,GPU=GPU))

def torch_inference_cpu(IMAGE,model):
    """Input: Image with shape: (3,n,m)"""
    model = model.cpu()
    mask = model(IMAGE.unsqueeze(0))
    p    = torch.functional.F.softmax(mask[0], 0)
    return p.argmax(0).cpu()

def inference_cpu(IMAGE,model):
    IMAGE = IMAGE.transpose((2,0, 1))
    IMAGE = (IMAGE / np.iinfo(IMAGE.dtype).max)
    IMAGE = torch.tensor(IMAGE, dtype=torch.float32)
    return np.asarray(torch_inference_cpu(IMAGE,model))

def get_leafs(mask,index=1):
    leafs = np.zeros_like(mask)
    leafs[mask==1] = 1
    return leafs

def mask_to_im(MASK):
    names  = ["background",
              "structure",
              "flycatcher",
              "background leafs",
              "background tomato",
              "foreground leafs",
              "foreground tomato"]

    colors = [[255,255,255],
              [180,180,180],
              [255,255,  0],
              [  0,128,  0],
              [128,  0,  0],
              [100,255,100],
              [255,100,100]]

    NM = np.ones((MASK.shape[0],MASK.shape[1],3))*255
    for i in range(len(colors)):
        if i in MASK:
            NM[MASK==i] = colors[i]
    return [colors, names, NM.astype("uint8")]


def LAI_from_video(path,outpath):

    try:
        outname = "/".join(path.split("/")[-3:])[:-4]+"/"
        out     = outpath+outname
        Path(out).mkdir(parents=True, exist_ok=True)

        #model_path = sorted(glob("weight_log/0000000027/*.pt"))[-1]
        model_path = "weights/all_200_leafs_ep_400.pt"
        model      = torch.load(model_path,map_location=torch.device('cpu'))
        #model      = UNET(3,3)
        #model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

        video = cv2.VideoCapture(path)
        fnr   = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        LEAFS = []
        lim   = 2000
        print("-----------------------------------")
        print("Nr. of frames: "+str(fnr)+" limited to: "+str(len(range(fnr)[:lim])))
        print("-----------------------------------")
        for i in tqdm(range(fnr)[:lim]):
            frame = cv2.cvtColor(video.read()[1],cv2.COLOR_BGR2RGB)
            mask  = inference_cpu(frame,model)
            leafs = get_leafs(mask)
            leafs = leafs[:leafs.shape[0]//2] ### only top half
            lpix  = len(leafs[leafs==1])/leafs.size
            LEAFS.append(lpix)
        video.release()
        LEAFS = np.asarray(LEAFS)

        np.save(out+"LAI",LEAFS)
    except:
        pass

if __name__ == '__main__':
    from MY_MODEL import UNET


    outpath    = "LAI_data_old_TH/"  ### only top half
    video_dic  = "/mnt/tomato_project/MIT_data/02_センシング/画像データ/"
    videos     = sorted(glob(video_dic+"**年/**/2**1/*.avi"))

    def cLAI(path):
        return LAI_from_video(path,outpath)

    mCPU(cLAI,videos,40)
