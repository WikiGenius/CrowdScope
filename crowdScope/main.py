# Author: Muhammed Elyamani
# Date: 03/02/2023
# GitHub: https://github.com/WikiGenius

from utils.layout import *
from conf import *
from utils import StyleApp
import asone
from asone import ASOne
import utils
import re
import time


class crowdScope(StyleApp):

    def on_start(self): 
        # Load YOLOv8n model for object detection
        print(f"load model: {model_path}")
        print(f"image size: {imgsz }")
        self.detector = ASOne(detector=asone.YOLOV8N_PYTORCH,weights=model_path ,use_cuda=True)

        self.ageNet=cv2.dnn.readNet(ageModel,ageProto)
        self.genderNet=cv2.dnn.readNet(genderModel,genderProto)
        
        self.pattern1 = re.compile(r'\d+')
        
    def on_stop(self):
        # Stop the detector when the app is closed
        pass  
    
    def analyse_image(self, frame):
        self.visualize = self.screen.vis.active

        process_time = time.time()
        frame_vis, faceBoxes = self.count_people(frame.copy())
        # frame_vis = self.analyse_faces(frame.copy(), frame_vis, faceBoxes)
        process_time = time.time() - process_time
        self.fps = 1 / process_time
        return frame_vis
    
        
    def analyse_button(self):
        if self.start == False: 
            self.start = True
            text = self.screen.analyse_button.text.replace("ANALYZE", "STOP")
            self.screen.analyse_button.text = text
            
        else:
            self.start = False
            text = self.screen.analyse_button.text.replace("STOP", "ANALYZE")
            self.screen.analyse_button.text = text
            self.stop_analyse()

    def count_people(self, frame):
        conf_thres = self.screen.conf_thres.value / 100
        iou_thres = self.screen.iou_thres.value / 100
        face_thres = self.screen.face_thres.value / 100
        dets, frame_info = self.detector.detect(frame, conf_thres=conf_thres, iou_thres=iou_thres, input_shape=imgsz)
        frame, count_people, faceBoxes = utils.draw_count_people(frame, dets, visualize=self.visualize, conf_thresh_face=face_thres )
        
        people_count_number = self.screen.people_count.text
        modified_people_count_number = self.pattern1.sub(f"{count_people}", people_count_number)
        self.screen.people_count.text = modified_people_count_number
        
        return frame, faceBoxes
    
    def analyse_faces(self, frame, frame_vis, faceBoxes):
        total_ages = 0
        total_genderList = []
        for faceBox in faceBoxes:
            face = utils.preprocess_face(frame, faceBox)

            
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            self.genderNet.setInput(blob)
            genderPreds=self.genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            self.ageNet.setInput(blob)
            agePreds=self.ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            
            total_genderList.append(gender)
            ag1, age2 = age.strip('()').split('-')
            total_ages += (int(ag1) + int(age2)) / 2
            
            if self.visualize:
                cv2.putText(frame_vis, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        
        if faceBoxes.shape[0] > 0:
            avg_age = int(total_ages / faceBoxes.shape[0])
            M_count = sum([1 for x in total_genderList if x=='Male' ])
            F_count = len(total_genderList) - M_count
            M_ratio = M_count /  len(total_genderList)
            F_ratio = F_count /  len(total_genderList)
        
        else:
            M_ratio = 0.5
            F_ratio = 0.5
            avg_age= 0
        if M_ratio < 0.3:
            self.screen.gender_M.text = ""
        else:
            self.screen.gender_M.text = "[font=Montserrat]M[/font]"
        if F_ratio < 0.3:
            self.screen.gender_F.text = ""
        else:
            self.screen.gender_F.text = "[font=Montserrat]F[/font]"
            
            
        self.screen.gender_M.size_hint_x =  M_ratio
        self.screen.gender_F.size_hint_x =  F_ratio
        
        avg_age_number = self.screen.avg_age.text
        modified_avg_age_number = self.pattern1.sub(f"{avg_age}", avg_age_number)
        self.screen.avg_age.text = modified_avg_age_number
        
        return frame_vis
    
    def process_after_video(self):
        self.stop_analyse()

    def stop_analyse(self):
        "=======================people_count================="
        people_count_number = self.screen.people_count.text
        modified_people_count_number = self.pattern1.sub(f"{0}", people_count_number)
        self.screen.people_count.text = modified_people_count_number
        "=======================avg_age======================"
        avg_age_number = self.screen.avg_age.text
        modified_avg_age_number = self.pattern1.sub(f"{0}", avg_age_number)
        self.screen.avg_age.text = modified_avg_age_number
        "=======================gender======================="
        self.screen.gender_M.size_hint_x =  0.5
        self.screen.gender_F.size_hint_x =  0.5
        self.screen.gender_M.text = "[font=Montserrat]M[/font]"
        self.screen.gender_F.text = "[font=Montserrat]F[/font]"
        "=======================FPS=========================="
        self.fps = 33
    

if __name__ == '__main__':
    crowdScope().run()
