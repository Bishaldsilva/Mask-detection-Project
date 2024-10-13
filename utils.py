import cv2 as cv
import numpy as np
from os.path import join
from tflite_runtime.interpreter import Interpreter

__model = None
__clf = None

color ={0:(0,255,0),1:(0,0,255), 2:(86, 13, 163)}
catg = ['Mask','No Mask', 'Wrong Mask']

def load_artifacts():
    global __model
    global __clf

    print(os.path.abspath(join('artifacts', 'model.tflite')))
    __model = Interpreter(model_path=join('artifacts', 'model.tflite'))
    __model.allocate_tensors()

    __clf = cv.CascadeClassifier(join('artifacts', 'haarcascade_frontalface_default.xml'))

load_artifacts()

# def classify(path):
#     image = cv.imread(path)
#     gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
#     faces = __clf.detectMultiScale(gray,1.3,4)
    
#     if len(faces) == 0:
#         cv.imwrite(join('images','result.jpg'),image)
#         return "No face Detected"

#     for x,y,w,h in faces:
#         f_img = image[y:y+h,x:x+w]
#     f_img = cv.resize(f_img,(50,50))
#     pred_case = __model.predict(np.array([f_img])/255)
    
#     n = [np.argmax(i) for i in pred_case]
#     cv.rectangle(image,(x,y),(x+w,y+h),color[n[0]],2)
#     cv.rectangle(image,(x,y-40),(x+w,y),color[n[0]],-1)
#     cv.putText(image,str(catg[n[0]]),(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
#     rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)
#     cv.imwrite(join('images','result.jpg'),image)

#     return f"Probability: {round(pred_case[0][n[0]] * 100, 2)}%"

def classify(path):
    image = cv.imread(path)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    faces = __clf.detectMultiScale(gray,1.3,4)
    
    if len(faces) == 0:
        cv.imwrite(join('images','result.jpg'),image)
        return "No face Detected"

    for x,y,w,h in faces:
        f_img = image[y:y+h,x:x+w]
    f_img = cv.resize(f_img,(50,50))
    f_img = np.array([f_img])/255

    # Set input tensor
    input_details = __model.get_input_details()
    __model.set_tensor(input_details[0]['index'], f_img.astype(np.float32))

    # Run inference
    __model.invoke()

    # Get output
    output_details = __model.get_output_details()
    output_data = __model.get_tensor(output_details[0]['index'])

    n = [np.argmax(i) for i in output_data]
    cv.rectangle(image,(x,y),(x+w,y+h),color[n[0]],2)
    cv.rectangle(image,(x,y-40),(x+w,y),color[n[0]],-1)
    cv.putText(image,str(catg[n[0]]),(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
    rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    cv.imwrite('/content/result.jpg',image)

    return f"Probability: {round(output_data[0][n[0]] * 100, 2)}%"

if __name__ == '__main__':
    load_artifacts()
    classify(join('images','test4.jpg'))