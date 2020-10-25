# -*- coding: utf-8 -*-
import wave
import sys
import numpy as np
import time
import os  
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    learning_wav_list = [f.name for f in os.scandir('../test_data/snap') if f.is_file()]
    print(learning_wav_list)
    for wav_file_name in learning_wav_list:
        data = parseWavFile('../test_data/snap/'+ wav_file_name)
        data = fft(data)
        addText(data, 'test_data.txt', 1)
    

# wavファイルを読み込んで正規化
# path : ファイル名を含む相対パス
def parseWavFile(path):
    wf = wave.open(path, 'r')
    channels = wf.getnchannels()
    chunk_size = wf.getnframes()
    amp = (2**8) ** wf.getsampwidth()/2
    data = wf.readframes(chunk_size)
    data = np.frombuffer(data, 'int16')
    data = data/amp
    data = data[::channels]
    wf.close()

    return data

# FFT 実部だけ返す　
# データ的に窓関数なし
def fft(data):
    x = np.fft.fft(data)
    return x.real[:int(len(x)/2)]

# 指定のファイルに
#データをカンマ区切り&末尾にタブ区切りでans(snap:1 nosnap:0) をつけて追記
# path : ファイル名を含む相対パス
def addText(data, path, ans):
    with open(path, 'a') as f:
        print(','.join(map(str,data))+'\t'+str(ans), file=f)

if __name__ == '__main__':
    main()
