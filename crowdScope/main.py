import os
from utils.layout import *
from utils import StyleApp

class crowdScope(StyleApp):
    
    def on_start(self): 
        # Load YOLOv8n model for object detection
        print(f"Is the model existed: {os.path.isfile(model_path)}")
         
    def on_stop(self):
        # Stop the detector when the app is closed
        pass  
    def analyse_image(self, frame):
        return frame  
    def analyse_button(self):
        print(f"ANALYZE... ")


if __name__ == '__main__':
    crowdScope().run()
