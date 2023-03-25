# Author: Muhammed Elyamani
# Date: 03/02/2023
# GitHub: https://github.com/WikiGenius

from conf import *
import torch
def predict_gender(face):
    results = gender_model.predict(face)[0]
    probs = results.probs
    idx = torch.argmax(probs).item()
    gender = GENDER_DICT[idx]
    return gender

def predict_age(face):
    results = age_model.predict(face)[0]
    probs = results.probs
    idx = torch.argmax(probs).item()
    age = AGEDICT[idx]
    return age




def predict_age_gender(face):
    gender = predict_gender(face)
    age = predict_age(face)
    
    return gender, age