# Import necessary packages
from kivy import platform
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivymd.uix.slider import MDSlider


model_path = ''

class Myslider(MDSlider):
    def on_touch_up(self, touch):
        self.active = True
        
# Define a custom widget for loading files
class LoadFile(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
        
# Define the main widget for the app
class MainScreen(Screen):
    pass       

    
