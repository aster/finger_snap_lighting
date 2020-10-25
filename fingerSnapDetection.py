# -*- coding: utf-8 -*-
#マイク0番からの入力を受ける。一定時間(RECROD_SECONDS)だけ録音し、ファイル名：mono.wavで保存する。
import pyaudio
import wave
import numpy as np
import joblib
import time
import subprocess
from make_data_set import wavDataProcess

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

    lampState = 0
    
    while True:
        data = stream.read(chunk)
        # nd.arrayに変換
        x = np.frombuffer(data, dtype="int16") / 32768.0

        if x.max() > threshold:
            # サンプル取り込み
            all = []
            all.append(data)
            data = b''.join(all)

            # 一旦出力
            out = wave.open('nowData.wav','w')
            out.setnchannels(CHANNELS) #mono
            out.setsampwidth(2) #16bits
            out.setframerate(RATE)
            out.writeframes(data)
            out.close()

            # 読み込み
            data = wavDataProcess.parseWavFile('nowData.wav')

            # fft
            data = wavDataProcess.fft(data)

            # SVM用に加工  指パッチンかどうかを判定（末尾1）
            data = ','.join(map(str,data))+'\t'+str(1)

            line=data.rstrip()

            tmp=line.split("\t")
            tmpx=tmp[0].split(",")
            tmpx=[float(j) for j in tmpx]
            tmpt=int(tmp[1])
            xtest,ttest=[],[]
            xtest.append(tmpx)
            ttest.append(tmpt)
            xtest=np.asarray(xtest,dtype=np.float32)
            ttest=np.asarray(ttest,dtype=np.int32)
     
            # 予測器読み込み
            predictor=joblib.load("svm_prd/predictor_svc.pkl")
            # 判定
            liprediction=predictor.predict(xtest) 

            if(liprediction[0]):
                print("snap!!!")
                if(lampState):  
                    p = subprocess.Popen(["tplight on 192.168.43.219"], shell=True)
                    time.sleep(2)
                    p.kill()

                else:
                    p = subprocess.Popen(["tplight off 192.168.43.219"], shell=True)
                    time.sleep(2)
                    p.kill()

                #オンオフでトグル
                lampState=~lampState

            else:
                print("oops")
        print('.')

    stream.close()    
    p.terminate()
