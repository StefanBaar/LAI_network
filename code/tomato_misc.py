from pathlib import Path

from tqdm import tqdm
from glob import glob

from datetime import datetime

import pandas as pd
import numpy as np
from scipy import ndimage
import cv2
from skimage import segmentation, morphology, transform, util

import xlrd

import subprocess as sp
from joblib import Parallel, delayed

def mCPU(func, var, n_jobs=20,verbose=10):
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(i) for i in var)


def date_from_ints(LIST):
    return [datetime.strptime(i, "%Y%m%d") for i in LIST]

def dates_to_days(DATE_LIST):
    D0       = DATE_LIST[0].replace(month=1,day=1)
    DAYS     = DATE_LIST-D0
    return np.asarray([i.days for i in DAYS]).astype(int)

def get_nan_inds(data):
    return np.arange(len(data))[np.invert(np.isnan(data))]

def flag_index(days,ind):
    mask      = np.ones_like(days)
    mask[ind] = 0
    mask      = mask.astype(bool)
    return days[mask]

### video stuff

def count_frames(path):
    video = cv2.VideoCapture(path)
    frames= int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frames

def count_frames_manual(video):
	# initialize the total number of frames read
	total = 0

	# loop over the frames of the video
	while True:
		# grab the current frame
		(grabbed, frame) = video.read()

		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break

		# increment the total number of frames read
		total += 1

	# return the total number of frames in the video file
	return total

def video_2_frames(inpath,video_dics,outpath):
    skipped = []
    for video in tqdm(video_dics):
        out = video.split(".avi")[0].replace(inpath,outpath)+"/"
        frames = count_frames(video)
        if frames > 100:
            Path(out).mkdir(parents=True, exist_ok=True)
            sp.call(["ffmpeg -i "+video.replace(" ","\ ")+" "+out.replace(" ","\ ")+"%06d.png"],shell=True)
        else:
            skipped.append(video)
    for i in skipped:
        print(i)
    np.save(skipped)


### excel stuff
def get_date(EXCEL_DATE):
    return datetime(*xlrd.xldate_as_tuple(EXCEL_DATE, 0))


def get_dates_by_year(dates,y=2020):
    db = []
    for i in dates:
        if str(y) == i[:4]:
            db.append(i)
    return np.asarray(db)

def corr_date(DATE_LIST):
    NEW_LIST = []
    for i in DATE_LIST:
        if isinstance(i, datetime) == False:
            NEW_LIST.append(get_date(i))
        else:
            NEW_LIST.append(i)
    return np.array(NEW_LIST)

def get_aru_mask(SRC,REF):
    src = [[i.month, i.day] for i in SRC]
    ref = [[i.month, i.day] for i in REF]

    MASK = []
    for i in ref:
        if i in src:
            MASK.append(1)
        else:
            MASK.append(0)
    return np.asarray(MASK)

def get_tomato_type(index_list):
    M1,M2 = [],[]
    for j,i in enumerate(index_list):
        if i[0] == "B":
            M1.append(j)
        else:
            M2.append(j)
    return np.asarray(M1), np.asarray(M2)


def get_2019_kind(path="/mnt/tomato_project/MIT_data/02_センシング/画像データ/2019年/動画ピックアップ/"):
    def get_dates(lists):
        return [i.split("/")[-1].split(".")[-2] for i in lists]
    dalts = np.unique(np.asarray(get_dates(glob(path+"ダルタリー/**/**/**/*.avi"))))
    tomis = np.unique(np.asarray(get_dates(glob(path+"富丸/**/**/**/*.avi"))))
    return [tomis,dalts]

def get_tomato_kind(path,DATES):
    ex2020             = pd.read_excel(path,engine='openpyxl')
    tomato_mask        = np.asarray(ex2020["Unnamed: 2"][1:].dropna())
    tom_mask, dal_mask = get_tomato_type(tomato_mask)
    return tom_mask, dal_mask

#### imaging

def get_green(frame, thresh = 125):
    GB   = (frame[:,:,1]-frame[:,:,0]).astype(float)
    GR   = np.abs(GB-frame[:,:,2]).astype(float)
    GRB   = ndimage.gaussian_filter(GR+GB,1)
    mask = np.zeros_like(GR).astype(int)
    mask[GRB < thresh] = 1
    #mask[GRB <      0] = 0
    mask = morphology.remove_small_objects(mask,
                                           min_size=1000,
                                           connectivity=morphology.disk(1))
    return mask

def outline(MASK, DISK = 3):
    OUTLINE = MASK.copy()
    return morphology.binary_dilation(OUTLINE,morphology.disk(DISK), out=None)-MASK

def color_overlay(image,mask,index = 1, DISK = 3):
    frame = image.copy()
    frame[:,:,1][np.where(mask == index)] += 50
    frame[:,:,0][np.where(mask == index)] += 30
    frame[:,:,2][np.where(mask == index)] += 10

    frame[:,:,0][np.where(outline(mask,DISK) == index)] = 0
    frame[:,:,1][np.where(outline(mask,DISK) == index)] = 0
    frame[:,:,2][np.where(outline(mask,DISK) == index)] = 0

    return frame

def wb(temp=3500):
    kelvin_table = {1000: (255,56,0),
                    1500: (255,109,0),
                    2000: (255,137,18),
                    2500: (255,161,72),
                    3000: (255,180,107),
                    3500: (255,196,137),
                    4000: (255,209,163),
                    4500: (255,219,186),
                    5000: (255,228,206),
                    5500: (255,236,224),
                    6000: (255,243,239),
                    6500: (255,249,253),
                    7000: (245,243,255),
                    7500: (235,238,255),
                    8000: (227,233,255),
                    8500: (220,229,255),
                    9000: (214,225,255),
                    9500: (208,222,255),
                    10000: (204,219,255)}

    return np.asarray(kelvin_table[temp])/255.

if __name__ == '__main__':

    colortemp = wb()

    in_path  = "/mnt/Local_SSD/stefan/noen_tomato/2021_04_09/GOPRO/images/GX010014/"
    out_path = "/mnt/Local_SSD/stefan/noen_tomato/2021_04_09/GOPRO/leafs/GX010014/"
    Path(out_path).mkdir(parents=True, exist_ok=True)

    in_files = sorted(glob(in_path+"*.png"))
    for i in tqdm(in_files):
        frame = cv2.cvtColor(cv2.imread(i),cv2.COLOR_BGR2RGB)
        WB    = frame.copy()
        for c in range(3):
            WB[:,:,c] = WB[:,:,c]*colortemp[c]
        mask  = get_green(WB, 120)
        image = color_overlay(frame,mask)
        cv2.imwrite(out_path+i.split("/")[-1],cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

##### video to frame (takes way too mauch memory)
#
#    video_dic  = "/mnt/tomato_project/MIT_data/02_センシング/画像データ/"
#    video_dics = sorted(glob(video_dic+"**年/**/2**/*.avi"))
#    outpath    = "/mnt/Local_SSD/stefan/tomato_vids/"
#    video_2_frames(video_dic,video_dics,outpath)




#####
