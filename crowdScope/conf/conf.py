# Author: Muhammed Elyamani
# Date: 03/02/2023
# GitHub: https://github.com/WikiGenius

import cv2
import asone
from ultralytics import YOLO

model_path = 'assets/weights/bestv4_n_640.pt'
detector = asone.ASOne(detector=asone.YOLOV8N_PYTORCH, weights=model_path ,use_cuda=True)
imgsz = int(model_path.split('.')[0].split('_')[-1])

# preprocess face
PADDING=0.1
FIX_SQUARE=True
EPS_SIZE = 0.05


gender_model_path = 'assets/weights/best_cls_gender.pt'
gender_model = YOLO(gender_model_path)  # load a pretrained YOLOv8n gender classification model


age_model_path = 'assets/weights/best_cls_age.pt'
age_model = YOLO(age_model_path)  # load a pretrained YOLOv8n age classification model


AGEDICT = {0: '(0-20)', 1: '(20-40)', 2: '(40-100)'}
GENDER_DICT = {0: 'F', 1: 'M'}



