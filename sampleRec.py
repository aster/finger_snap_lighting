# -*- coding: utf-8 -*-
#マイク0番からの入力を受ける。一定時間(RECROD_SECONDS)だけ録音し、ファイル名：mono.wavで保存する。
import pyaudio
import wave
import sys
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    #サンプリングレート、マイク性能に依存
    RATE = 8000
    #録音時間
    RECORD_SECONDS = 1

    #閾値
    threshold = 0.5

    #pyaudio
    p = pyaudio.PyAudio()
    #マイク0番を設定
    input_device_index = 0
    #マイクからデータ取得
    stream = p.open(format = FORMAT,
            channels = CHANNELS,
            rate = RATE,
            input = True,
            frames_per_buffer = chunk
            )

    cnt = 0
    while True:
        data = stream.read(chunk)
        # nd.arrayに変換
        x = np.frombuffer(data, dtype="int16") / 32768.0

        if x.max() > threshold:
            filename = datetime.today().strftime("%Y%m%d%H%M%S") + ".wav"
            print(filename)

            # サンプル取り込み
            all = []
            all.append(data)
            # 1024サンプル
            #data = stream.read(chunk)
            #all.append(data)
            data = b''.join(all)                    

            # ファイル出力
            out = wave.open('training/nosnap/'+filename,'w')
            out.setnchannels(CHANNELS) #mono
            out.setsampwidth(2) #16bits
            out.setframerate(RATE)
            out.writeframes(data)
            out.close()

            # ちゃんと波形出てるかチェック
            # glaph start --------------------
            x = np.frombuffer(data, dtype="int16") / 32768.0

            plt.figure(figsize=(15,3))
            plt.plot(x)
            plt.savefig("wave.png")

            x = np.fft.fft(np.frombuffer(data, dtype="int16"))
            freq = np.fft.fftfreq(int(len(x)), 1/8000)
            max_freq = int(len(x))/2

            plt.figure(figsize=(15,3))
            plt.plot(x.real[:int(len(x)/2)])
            plt.savefig("fft.png")
            # glaph end ---------------

            cnt+=1
            print("Saved. cnt=", cnt)

    stream.close()    
    p.terminate()
