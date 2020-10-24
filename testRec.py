# -*- coding: utf-8 -*-
#マイク0番からの入力を受ける。一定時間(RECROD_SECONDS)だけ録音し、ファイル名：mono.wavで保存する。
import pyaudio
import sys
import time
import wave

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    #サンプリングレート、マイク性能に依存
    RATE = 8000
    #録音時間
    RECORD_SECONDS =int(input('Please input recoding time>>>'))

    #pyaudio
    p = pyaudio.PyAudio()

    #マイク0番を設定
    input_device_index = 0
    #マイクからデータ取得
    stream = p.open(format = FORMAT,
            channels = CHANNELS,
            rate = RATE,
            input = True,
            frames_per_buffer = chunk)
    all = []
    for i in range(0, int(RATE / chunk * RECORD_SECONDS)):
        data = stream.read(chunk)
        all.append(data)

    stream.close()    
    data = b''.join(all)                    

    # glaph start --------------------
    x = np.frombuffer(data, dtype="int16") / 32768.0

    plt.figure(figsize=(15,3))
    plt.plot(x)
    plt.show()

    x = np.fft.fft(np.frombuffer(data, dtype="int16"))
    freq = np.fft.fftfreq(int(len(x)), 1/8000) 
    max_freq = int(len(x))/2

    plt.figure(figsize=(15,3))
    plt.plot(x.real[:int(len(x)/2)])
    plt.show()

    # glaph end ---------------

    out = wave.open('mono.wav','w')
    out.setnchannels(1) #mono
    out.setsampwidth(2) #16bits
    out.setframerate(RATE)
    out.writeframes(data)
    out.close()

    p.terminate()
