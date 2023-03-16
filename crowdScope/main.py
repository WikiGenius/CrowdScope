import os
from utils.layout import *
from utils import StyleApp
import asone
from asone import ASOne
import utils
import re
import time

# gender_M:gender_M
# gender_F:gender_F
# avg_age:avg_age
# People_count:People_count

class crowdScope(StyleApp):
    
    def on_start(self): 
        # Load YOLOv8n model for object detection
        self.detector = ASOne(detector=asone.YOLOV8N_PYTORCH, use_cuda=True)
        self.pattern1 = re.compile(r'\d+')
        
    def on_stop(self):
        # Stop the detector when the app is closed
        pass  
    
    def analyse_image(self, frame):
        frame = self.count_people(frame)
        return frame
    
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
        dets, frame_info = self.detector.detect(frame, conf_thres=0.25, iou_thres=0.45)
        if dets is not None: 
            bbox_xyxy = dets[:, :4]
            scores = dets[:, 4]
            class_ids = dets[:, 5]
            frame, count_people = utils.count_people(frame, bbox_xyxy, class_ids=class_ids)
            
            people_count_number = self.screen.people_count.text
            modified_people_count_number = self.pattern1.sub(f"{count_people}", people_count_number)
            self.screen.people_count.text = modified_people_count_number
            
            print(count_people)
        return frame
    
    def process_after_video(self):
        self.stop_analyse()

    def stop_analyse(self):
        "=======================people_count================="
        people_count_number = self.screen.people_count.text
        modified_people_count_number = self.pattern1.sub(f"{0}", people_count_number)
        self.screen.people_count.text = modified_people_count_number
        "====================================================="
    
        

if __name__ == '__main__':
    crowdScope().run()
