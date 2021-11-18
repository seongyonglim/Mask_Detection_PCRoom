#!/usr/bin/env python
# coding: utf-8

# In[2]:



import os
import platform
from IPython.display import clear_output
from IPython.display import HTML
from base64 import b64encode
from tensorflow.keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import warnings
warnings.filterwarnings("ignore")
from google.colab.patches import cv2_imshow
from google.colab import drive
drive.mount('/content/drive')

# 테스트 영상을 해당 화면에서 출력하기 위한 함수
def play(filename):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=640 muted controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)


# ## **Testing**

# In[15]:


# 테스트 진행에 필요한 파일들의 경로를 설정함
args = { 

    "model": "/content/drive/My Drive/km/model/model_v1.h5",
    "label-bin": "/content/drive/My Drive/km/model/lb.pickle",
    "input": "/content/drive/My Drive/km/example_clips/lifting.mp4",
    "size" : 128,
    "output" : "/content/drive/My Drive/km/output/V_2.avi"   
}


# ## **Input** : 입력할 테스트 영상 확인

# In[16]:


mask=args['input'] 
play(mask)


# ## **Output**

# ## 1. 프레임 별로 어떻게 모델이 예측 결과를 나타내는 지 확인

# In[17]:


vs = cv2.VideoCapture(mask)
writer = None
(W, H) = (None, None)

model = load_model(args['model'])
lb = pickle.loads(open(args['label-bin'], "rb").read())
count = 0  


Q=[]
while True:
	# 테스트 영상으로 부터 한 개의 프레임 불러오기
	(grabbed, frame) = vs.read()

	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]
  # 불러온 해당 프레임의 사이즈를 조정하고, 스케일링 진행
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	output = cv2.resize(frame, (512, 360)).copy()
	frame = cv2.resize(frame, (128, 128)).astype("float32")
	frame = frame.reshape(128, 128, 3) / 255
	# 기존에 학습한 우리의 모델에 입력하여 마스크 착용 여부의 예측값을 저장
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)
	results = np.array(Q).mean(axis=0)
	i = 1
	label = lb.classes_[i]
	

	# 해당 프레임마다 모델의 예측 결과를 텍스트로 채워주기
	text_color = (0, 255, 0) # default : green

	if preds[0] > 0.50 : # mask prob
		text_color = (0, 0, 255) # red
		label = 'wearing a mask'
	else:
		label = 'Not wearing a mask'

	text = "State : {:8} ({:3.2f}%)".format(label,preds[0]*100)
	FONT = cv2.FONT_HERSHEY_SIMPLEX 

	cv2.putText(output, text, (35, 50), FONT,1.25, text_color, 3) 
	output = cv2.rectangle(output, (35, 80), (35+int(preds[0])*5,80+20), text_color,-1)

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,(W, H), True)
	writer.write(output)

	# 프레임 이미지를 출력하기
	cv2_imshow(output)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

print("[INFO] cleaning up...")
writer.release()


# ## 2. 최종 해당 영상에 대해 착용 여부 결과 확인

# In[18]:


# 테스트할 영상을 프레임 단위로 불러온 후 예측을 진행한다.
vs = cv2.VideoCapture(Violence)
writer = None
(W, H) = (None, None)

model = load_model(args['model'])
lb = pickle.loads(open(args['label-bin'], "rb").read())
count = 0 

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # 영상의 하나의 프레임을 불러온 후 전처리를 진행한다.

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = cv2.resize(frame, (512, 360)).copy()
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame = frame.reshape(128, 128, 3) / 255
    # 예측을 진행하고 리스트에 저장
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# In[1]:


# 프레임 단위로 예측된 결과들을 취합하여, 최종 착용 여부를 판단한다.
results = np.array(Q).mean(axis=0)

prob = round(results[0]*100,2)
if prob >= 50:
    label = 'wearing a mask'
else:
    label = 'Not wearing a mask' 

print('해당 영상에 대한 모델의 예측 결과 : \n')
print('해당 영상은 {} 클래스에 속합니다.'.format(label))


# In[ ]:




