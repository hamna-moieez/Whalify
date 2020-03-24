import glob 
import os
import wave
from tqdm import tqdm 
import pylab
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

dataset_ = "/content/drive/My Drive/ResearchWork/dataset/"
rem_list = ['0', '1']

def graph_spectrogram(wav_file, class_, filename):
    sound_info, frame_rate = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.savefig(dataset_+'{}.png'.format(filename))
    pylab.close()

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

for j in rem_list:
    for i in tqdm(glob.glob(dataset_+'{}/*.wav'.format(j))):
        file_name = i.split('/')[-1]
        class_info = i.split('/')[-2]
        graph_spectrogram(i, class_info, file_name)
