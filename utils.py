import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2 as cv
import numpy as np
from os.path import join
import os
import gdown

__model = None
__clf = None

color ={0:(0,255,0),1:(0,0,255), 2:(86, 13, 163)}
catg = ['Mask','No Mask', 'Wrong Mask']

FILE_ID = '1n44gIe3ZnWW9ukfNXIHauZCN9ObCSb5D'
MODEL_FILE_NAME = 'mask_detection_model_2.h5'
LOCAL_MODEL_PATH = f'./artifacts/{MODEL_FILE_NAME}'
GDRIVE_URL = f'https://drive.google.com/uc?id={FILE_ID}'

def download_model_from_gdrive():
    if not os.path.exists(LOCAL_MODEL_PATH):
        gdown.download(GDRIVE_URL, LOCAL_MODEL_PATH, quiet=False)

def load_artifacts():
    global __model
    global __clf

    download_model_from_gdrive()

    __model = load_model(join('artifacts', 'mask_detection_model_2.h5'))
    __clf = cv.CascadeClassifier(join('artifacts', 'haarcascade_frontalface_default.xml'))

load_artifacts()

def classify(path):
    image = cv.imread(path)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    faces = __clf.detectMultiScale(gray,1.3,4)
    
    if len(faces) == 0:
        cv.imwrite(join('images','result.jpg'),image)
        return "No face Detected"

    for x,y,w,h in faces:
        f_img = image[y:y+h,x:x+w]
    f_img = cv.resize(f_img,(128,128))
    pred_case = __model.predict(np.array([f_img])/255)
    
    n = [np.argmax(i) for i in pred_case]
    cv.rectangle(image,(x,y),(x+w,y+h),color[n[0]],2)
    cv.rectangle(image,(x,y-40),(x+w,y),color[n[0]],-1)
    cv.putText(image,str(catg[n[0]]),(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
    rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    cv.imwrite(join('images','result.jpg'),image)

    return f"Probability: {round(pred_case[0][n[0]] * 100, 2)}%"

if __name__ == '__main__':
    load_artifacts()
    classify(join('images','test4.jpg'))