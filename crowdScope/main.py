import os
from utils.layout import *
from utils import StyleApp
import asone
from asone import ASOne
from asone import utils
import time
class crowdScope(StyleApp):
    
    def on_start(self): 
        # Load YOLOv8n model for object detection
        self.detector = ASOne(detector=asone.YOLOV8N_PYTORCH, weights=model_path, use_cuda=True)
        
    def on_stop(self):
        # Stop the detector when the app is closed
        pass  
    
    def analyse_image(self, frame):
        dets, frame_info = self.detector.detect(frame, conf_thres=0.25, iou_thres=0.45)
        if dets is not None: 
            bbox_xyxy = dets[:, :4]
            scores = dets[:, 4]
            class_ids = dets[:, 5]
            frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids, class_names=['face'])
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

            

if __name__ == '__main__':
    crowdScope().run()
