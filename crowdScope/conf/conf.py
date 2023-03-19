
visualize = True
iou_thres=0.45

# model_path = 'assets/weights/best_s_640.pt'
# model_path = 'assets/weights/best_s_928.pt'
model_path = 'assets/weights/best_s_640.pt'

imgsz = int(model_path.split('.')[0].split('_')[-1])




ageProto="assets/weights/age_deploy.prototxt"
ageModel="assets/weights/age_net.caffemodel"

genderProto="assets/weights/gender_deploy.prototxt"
genderModel="assets/weights/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

padding=20