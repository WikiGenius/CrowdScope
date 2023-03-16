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

model_path = 'assets/weights/best.pt'

# importing labelbase which
# register our custom font for application
from kivy.core.text import LabelBase
# registering our new custom fontstyle
LabelBase.register(name='Montserrat',
                   fn_regular='assets/fonts/Montserrat/static/Montserrat-Black.ttf')

# Define a custom widget for loading files
class LoadFile(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
        
# Define the main widget for the app
class MainScreen(Screen):
    pass       

    
