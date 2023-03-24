import cv2
import asone
from ultralytics import YOLO

# model_path = 'assets/weights/best_n_640.pt'
# model_path = 'assets/weights/best_n__928.pt'
# model_path = 'assets/weights/best_s_640.pt'
# model_path = 'assets/weights/bestv2_n_640.pt'
# model_path = 'assets/weights/bestv3_n_640.pt'
model_path = 'assets/weights/bestv4_n_640.pt'
# model_path = 'assets/weights/bestv5_n_640.pt'

detector = asone.ASOne(detector=asone.YOLOV8N_PYTORCH,weights=model_path ,use_cuda=True)


ageProto="assets/weights/age_deploy.prototxt"
ageModel="assets/weights/age_net.caffemodel"

genderProto="assets/weights/gender_deploy.prototxt"
genderModel="assets/weights/gender_net.caffemodel"

ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

imgsz = int(model_path.split('.')[0].split('_')[-1])

# preprocess face
PADDING=0.1
FIX_SQUARE=True
EPS_SIZE = 0.05


gender_model_path = 'assets/weights/best_cls_gender.pt'
gender_model = YOLO(gender_model_path)  # load a pretrained YOLOv8n classification model

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

GENDER_DICT={0: 'F', 1: 'M'}



