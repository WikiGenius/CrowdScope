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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.imgsz = 640
        self.imgsz = int(model_path.split('.')[0].split('_')[-1])
        self.visualize = True
        self.iou_thres=0.45
    def on_start(self): 
        # Load YOLOv8n model for object detection
        print(f"load model: {model_path}")
        print(f"image size: {self.imgsz }")
        self.detector = ASOne(detector=asone.YOLOV8N_PYTORCH,weights=model_path ,use_cuda=True)
        self.pattern1 = re.compile(r'\d+')
        
    def on_stop(self):
        # Stop the detector when the app is closed
        pass  
    
    def analyse_image(self, frame):
        process_time = time.time()
        frame, faces = self.count_people(frame)
        frame = self.analyse_faces(frame, faces)
        process_time = time.time() - process_time
        self.fps = 1 / process_time
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
        conf_thres = self.screen.conf_thres.value / 100
        dets, frame_info = self.detector.detect(frame, conf_thres=conf_thres, iou_thres=self.iou_thres, input_shape=self.imgsz)
        frame, count_people, faces = utils.count_people(frame, dets, visualize=self.visualize )
        people_count_number = self.screen.people_count.text
        modified_people_count_number = self.pattern1.sub(f"{count_people}", people_count_number)
        self.screen.people_count.text = modified_people_count_number
            
        return frame, faces
    def analyse_faces(self, frame, faces):
        avg_age_number = self.screen.avg_age.text
        modified_avg_age_number = self.pattern1.sub(f"{faces.shape[0]}", avg_age_number)
        self.screen.avg_age.text = modified_avg_age_number
        
        return frame
    
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
        "=======================FPS=========================="
        self.fps = 33
        

if __name__ == '__main__':
    crowdScope().run()
