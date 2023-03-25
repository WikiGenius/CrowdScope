# Author: Muhammed Elyamani
# Date: 03/02/2023
# GitHub: https://github.com/WikiGenius

from utils.layout import *
from conf import *
from utils import StyleApp
import utils
import re
import time


class crowdScope(StyleApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load YOLOv8n model for object detection
        print(f"load model: {model_path}")
        print(f"image size: {imgsz }")
        self.pattern = re.compile(r'\d+')
        
    def on_start(self): 
        self.process = utils.Process(self.screen, self.pattern) 

        
    def on_stop(self):
        # Stop the detector when the app is closed
        pass  
    
    def analyse_image(self, frame):
        self.process.visualize = self.screen.vis.active

        process_time = time.time()
        frame_vis, faceBoxes = self.process.count_people(frame.copy())
        frame_vis = self.process.analyse_faces(frame.copy(), frame_vis, faceBoxes)
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

 
    def process_after_video(self):
        self.stop_analyse()

    def stop_analyse(self):
        "=======================people_count================="
        people_count_number = self.screen.people_count.text
        modified_people_count_number = self.pattern.sub(f"{0}", people_count_number)
        self.screen.people_count.text = modified_people_count_number
        "=======================avg_age======================"
        avg_age_number = self.screen.avg_age.text
        modified_avg_age_number = self.pattern.sub(f"{0}", avg_age_number)
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
