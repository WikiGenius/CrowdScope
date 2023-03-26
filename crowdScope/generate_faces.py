import cv2
from conf import *
import asone
from asone import ASOne
import utils
import re
import time
import os
# Load YOLOv8n model for object detection
print(f"load model: {model_path}")
print(f"image size: {imgsz }")
detector = ASOne(detector=asone.YOLOV8N_PYTORCH,weights=model_path ,use_cuda=True)

dataset = './assets/generate_faces'
PREPROCESS = True
try:
    os.makedirs(dataset)
except FileExistsError:
    print("Directory already exists")


preprocess_path = './assets/preprocess/'
try:
    os.makedirs(preprocess_path)
except FileExistsError:
    print("Directory already exists")


    
cap=cv2.VideoCapture('./assets/videos/video_rec.mkv')

def fix_square(faceBox):
    w = faceBox[2] - faceBox[0]
    h = faceBox[3] - faceBox[1]
    print(f"before fixing square: {w} X {h}")
    
    padding_fix_square = abs(w - h) // 2
    if w < h:
        faceBox[0] -= padding_fix_square
        faceBox[2] += padding_fix_square
    elif w > h:
        faceBox[1] -= padding_fix_square
        faceBox[3] += padding_fix_square
    
    w = faceBox[2] - faceBox[0]
    h = faceBox[3] - faceBox[1]
    print(f"after fixing square: {w} X {h}")

    return faceBox, w

def preprocess_face(frame, faceBox):
    if FIX_SQUARE:
        faceBox, w = fix_square(faceBox)
        
    padding = int(PADDING * w)
    
    face=frame[max(0,faceBox[1]-padding):
           min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
           :min(faceBox[2]+padding, frame.shape[1]-1)]
    print(f"shape after padding: {face.shape}")
    face = cv2.resize(face, (128, 128))
    print(f"new shape: {face.shape}")
    print('===============================================')
    return face

count_faces = 0
def generate_faces( frame, faceBoxes):
    global count_faces
    for faceBox in faceBoxes:
        face = preprocess_face(frame, faceBox)
        if PREPROCESS:
            cv2.imwrite(f"{preprocess_path}/face_{face.shape[:-1]}_{str(count_faces).zfill(4)}.jpg", face)  
        count_faces+=1

def count_people( frame):
        resultImg = frame.copy()
        dets, frame_info = detector.detect(resultImg, conf_thres=0.25, iou_thres=iou_thres, input_shape=imgsz)
        resultImg, count_people, faceBoxes = utils.draw_count_people(resultImg, dets, visualize=visualize )
        return resultImg, faceBoxes

# loop runs if capturing has been initialized. 
while(1): 
  
    # reads frames from a video 
    ret, frame = cap.read() 
    if not ret:
        break
    _, faceBoxes = count_people( frame.copy())
    generate_faces(frame.copy(), faceBoxes)
    
    # Wait for 'q' to stop the program 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# Close the window / Release webcam
cap.release() 
  
# De-allocate any associated memoryusage 
cv2.destroyAllWindows()
    