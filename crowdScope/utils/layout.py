# Import necessary packages
from kivy import platform
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivymd.uix.slider import MDSlider
import cv2
from kivymd.app import MDApp
from kivymd.theming import ThemeManager
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from utils import resize, create_rounded_img
from plyer import filechooser

# model_path = 'assets/weights/best_s_640.pt'
# model_path = 'assets/weights/best_s_928.pt'
model_path = 'assets/weights/best_s_640.pt'

ageProto="assets/weights/age_deploy.prototxt"
ageModel="assets/weights/age_net.caffemodel"
genderProto="assets/weights/gender_deploy.prototxt"
genderModel="assets/weights/gender_net.caffemodel"
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
padding=20


# importing labelbase which
# register our custom font for application
from kivy.core.text import LabelBase
# registering our new custom fontstyle
LabelBase.register(name='Montserrat',
                   fn_regular='assets/fonts/Montserrat/static/Montserrat-Black.ttf')


class Myslider(MDSlider):
    def on_touch_up(self, touch):
        self.active = True

# Define the main widget for the app
class MainScreen(Screen):
    pass       

    
