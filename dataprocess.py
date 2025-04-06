import os
import subprocess
import pyworld as pw
import librosa
import matplotlib.pyplot as plt
import numpy as np
from textgrid import TextGrid
import pandas as pd


def gets_mean_pitch(wav_path):

    x, fs = librosa.load(wav_path)
    x = x.astype(np.double)
    _f0_h, t_h = pw.dio(x, fs)

    f0_h = pw.stonemask(x, _f0_h, t_h, fs)
    data = f0_h
    mask = (data != 0)
    segments = np.split(data, np.where(np.diff(mask))[0]+1)
    non_zero_num,value = 0 ,0
    for seg in segments:
        if len(seg)<=0 or seg[0]==0:
            continue
        non_zero_num += len(seg)
        value += seg.sum()
    if non_zero_num==0:
      return 0,False
    return value/non_zero_num,True

def get_energy(wav_path):
    y, sr = librosa.load(wav_path)
    energy = librosa.feature.rms(y=y)
    energy = energy.mean()
    return energy

def extract_tempo(textgrid_fp):
    itvs = TextGrid.fromFile(textgrid_fp)[0]
    num =0
    dur =0
    for i in range(len(itvs)):
        if itvs[i].mark == '':
            continue
        num += 1
        dur += itvs[i].maxTime - itvs[i].minTime
    tempo = dur /num


