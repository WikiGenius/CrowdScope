from conf import *

def predict_gender(blob):

    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    return gender

def predict_age(blob):
    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]
    return age




def predict_age_gender(face):
    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    gender = predict_gender(blob)
    age = predict_age(blob)
    
    return gender, age