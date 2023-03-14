import os
import cv2
from kivymd.app import MDApp
from kivymd.theming import ThemeManager
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from utils import resize, create_rounded_img
from utils.layout import *
from plyer import filechooser
from kivy.properties import ListProperty



class crowdScope(MDApp):
    selection = ListProperty([])
    theme_cls = ThemeManager()
    def build(self):
        self.theme_cls.theme_style = "Dark"

        self.started = False
        self.fps = 33

        # Create instance of SearchDashboard
        self.screen = MainScreen(name='main')

        # Return the instance of SearchDashboard
        return self.screen
    
    def on_start(self): 
        self.thread = True

        # Load YOLOv8n model for object detection
        print(f"Is the model existed: {os.path.isfile(model_path)}")
        # Initialize variables for video capture
        self.frame_count = 0
        # Initialize variable for filtering object classes
        self.filter_classes = None 

         
    def on_stop(self):
        # Stop the detector when the app is closed
        pass  
        
    def update(self, *args):
        # Read a frame from the video capture device
        ret, frame = self.capture.read()
        # Stop the detector if there are no more frames
        if not ret:
            # self.screen.detection_image.source = './data/upload.png'
            self.screen.detection_image.opacity = 0.1
            return
        frame = resize(frame, height=600)
    
        # Perform object detection on the frame using the YOLOv8n model
        # print(self.screen.slider.value)
        
        frame = create_rounded_img(frame, border_radius=40)
        
        if self.thread:
            cv2.line(frame, (20, 25), (127, 25), [85, 45, 255], 30)
            cv2.putText(frame, f'FPS: {int(self.fps)}', (11, 35), 0, 1, [
                    225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        # Flip the frame vertically for display purposes
        buf = cv2.flip(frame, 0).tobytes()
        # Create a Kivy Texture from the frame
        
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            

        # Update the image in SearchDashboard with the new frame
        self.screen.detection_image.texture = img_texture
        self.screen.detection_image.opacity = 1
        # Increment frame count
        self.frame_count += 1
        # Clock.schedule_once(self.update)
        
    def find_object(self):
        print(f"Finding object... {self.screen.search_input.text}")
        # Update the filter_classes variable based on the text input in SearchDashboard
        self.filter_classes = self.screen.search_input.text
        if self.filter_classes:
            self.filter_classes = self.filter_classes.split(',')    
            if platform == 'android':
                self.fps = 2
            else:
                self.fps = 20
        else:
            self.fps = 33
            
    def upload_video(self):
        '''
        Call plyer filechooser API to run a filechooser Activity.
        '''
        filters = [ '*.mp4']  # add more video file extensions here
        path = './assets/videos/*'
        filechooser.open_file(filters=filters, on_selection=self.handle_selection, path=path)

    def handle_selection(self, selection):
        '''
        Callback function for handling the selection response from Activity.
        '''
        if selection is not None:
            self.selection = selection

    def on_selection(self, *a, **k):
        '''
        Update TextInput.text after FileChoose.selection is changed
        via FileChoose.handle_selection.
        '''
        self.selected_file = True
        print("Uploading video...")
        # Load the selected video file
        self.start_app(self.selection[0])

    def start_app(self, vid_path):
        # Create a video capture object for the selected video file
        print(f'load file: {vid_path}')
        print(f'file exist: {os.path.isfile(vid_path)}')
        self.capture = cv2.VideoCapture(vid_path)
        # Schedule the update function to be called at 33 FPS
        Clock.schedule_interval(self.update, 1/self.fps)
        # Clock.schedule_once(self.update)

if __name__ == '__main__':
    crowdScope().run()
