from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.clock import Clock
from kivy.core.window import Window
import cv2
import os
import numpy as np
from objectloader import OBJ
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from render import *

ERROR_DIST = 150
MIN_MATCHES = 120
DEFAULT_COLOR = (0, 0, 255)
TARGET = ""
models = ['H2O.jpg', 'NaCl.jpg', 'FePt.jpg']
materials = ['H2O.mtl', 'H2O.mtl', 'H2O.mtl']

current_target = None

model_name = None

class MainScreen(Screen):
    def on_pre_enter(self):
        layout = MDBoxLayout(orientation='vertical', padding=20, spacing=20)
        self.add_widget(layout)

    def select_target(self, instance):
        #global TARGET
        selected = instance.text
        if selected == 'Câmera Inteligente':
            self.manager.current = 'camera'
        else:
            self.manager.current = 'biblioteca'
        #self.manager.current = 'camera'

class BibliotecaScreen(Screen):
    def on_pre_enter(self):
        layout = MDBoxLayout(orientation='vertical', padding=20, spacing=20)

        for model in models:
            btn = MDRaisedButton(
                text=model.split('.')[0], 
                on_release=self.select_target,
                halign='center',
                valign='center',
            )
            layout.add_widget(btn)

        self.add_widget(layout)

    def select_target(self, instance):
        #global TARGET
        selected = instance.text
        '''
        if selected == 'Câmera Inteligente':
            self.manager.current = 'camera'
        else:
            self.manager.current = 'biblioteca'
        '''

        print(f"Selected: {selected}")
        global model_name
        model_name = selected

        self.manager.current = 'cameraespecifica'
        
        #self.manager.current = 'camera'

'''
eclass MenuScreen(Screen):

    def on_pre_enter(slf):
        layout = MDBoxLayout(orientation='vertical', padding=20, spacing=20)
        self.add_widget(layout)

    def on_pre_enter(self):
        layout = MDBoxLayout(orientation='vertical', padding=20, spacing=20)
        for model in models:
            btn = MDRaisedButton(text=model, on_release=self.select_target)
            layout.add_widget(btn)
        self.add_widget(layout)

    def select_target(self, instance):
        global TARGET
        TARGET = instance.text
        self.manager.current = 'camera'
'''

class CameraScreen(Screen):
    def on_enter(self):
        self.capture = cv2.VideoCapture(0)
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        dir_name = os.getcwd()
        self.objNaCl = OBJ(os.path.join(dir_name, 'models/nacl.obj'), swapyz=True)
        self.objH2O = OBJ(os.path.join(dir_name, 'models/H2O.obj'), swapyz=True)
        self.objFePt = OBJ(os.path.join(dir_name, 'models/FePt.obj'), swapyz=True)
        self.camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        self.model_data = {}
        self.current_target = ''

        #model_data = {}
        for model_name in models:
            model_img = cv2.imread('./targets/'+model_name, 0)
            if model_img is None:
                print(f"Error loading model image: {model_name}")
                continue
            kp_model, des_model = self.orb.detectAndCompute(model_img, None)
            self.model_data[model_name] = (kp_model, des_model, model_img)

        # Create an Image widget to display the camera feed
        self.img = Image()
        self.add_widget(self.img)

        self.update()

    def update(self, *args):
        self.current_target = TARGET

        ret, frame = self.capture.read()
        if ret:
            # Convert the frame to texture
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture

            frame = cv2.resize(frame, (640, 480))
            self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Process frame and render OBJ model
            # (Add your frame processing and rendering code here)
            max_matches = 0
            new_target = None

            kp_frame, des_frame = self.orb.detectAndCompute(self.gray_frame, None)

            if des_frame is None:
                cv2.imshow('Camera', frame)

            for model_name, (kp_model, des_model, model_img) in self.model_data.items():
                matches = self.bf.match(des_model, des_frame)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [m for m in matches if m.distance < ERROR_DIST]

                #print(f"Model: {model_name}, Matches: {len(good_matches)}")

                if len(good_matches) > max_matches:
                    max_matches = len(good_matches)
                    new_target = model_name
                    good_matches_for_homography = good_matches

            if max_matches > MIN_MATCHES and new_target:
                if new_target != self.current_target:
                    self.current_target = new_target
                    kp_model, des_model, model_img = self.model_data[self.current_target]

                src_pts = np.float32([kp_model[m.queryIdx].pt for m in good_matches_for_homography]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches_for_homography]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                #print(f"Matches: {len(good_matches)}")

                if M is not None:
                    try:
                        projection = projection_matrix(self.camera_parameters, M)
                        mainObj = None

                        match(self.current_target):
                            case 'NaCl.jpg':
                                mainObj = self.objNaCl
                            case 'FePt.jpg':
                                mainObj = self.objFePt
                            case 'H2O.jpg':
                                mainObj = self.objH2O
                            case _:
                                print("Modelo não reconhecido")


                        print(f"Model: {self.current_target}, Matches: {len(good_matches)}")
                        #sleep(1)
                        frame = render(frame, mainObj, projection)
                        print(f'Projection: {projection}')
                    except Exception as e:
                        print(f"Render error: {e}")

        #cv2.imshow('Camera', frame)
        self.display_frame(frame)

        self.schedule = Clock.schedule_once(self.update, 1.0 / 30.0)
        #self.schedule = Clock.schedule_once(self.update, 2.0 / 30.0)

    def display_frame(self, frame):
        # Convert the frame to texture
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture

    def on_leave(self):
        self.capture.release()
        Clock.unschedule(self.schedule)

class CameraEspecificaScreen(Screen):
    def on_enter(self):
        self.capture = cv2.VideoCapture(0)
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        dir_name = os.getcwd()
        self.objNaCl = OBJ(os.path.join(dir_name, 'models/nacl.obj'), swapyz=True)
        self.objH2O = OBJ(os.path.join(dir_name, 'models/H2O.obj'), swapyz=True)
        self.objFePt = OBJ(os.path.join(dir_name, 'models/FePt.obj'), swapyz=True)
        self.camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        self.model_data = {}
        self.current_target = ''
        self.reavaliate = False
        '''self.vddMmuda = 0
        self.vddM = None'''

        #model_data = {}
        for model_name in models:
            model_img = cv2.imread('targets/'+model_name, 0)
            if model_img is None:
                print(f"Error loading model image: {model_name}")
                continue
            kp_model, des_model = self.orb.detectAndCompute(model_img, None)
            self.model_data[model_name] = (kp_model, des_model, model_img)

        # Create an Image widget to display the camera feed
        self.img = Image()
        self.add_widget(self.img)

        self.update()

    def update(self, *args):
        self.current_target = TARGET

        ret, frame = self.capture.read()
        if ret:
            # Convert the frame to texture
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture

            frame = cv2.resize(frame, (640, 480))
            self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Process frame and render OBJ model
            # (Add your frame processing and rendering code here)
            max_matches = 0
            new_target = None

            kp_frame, des_frame = self.orb.detectAndCompute(self.gray_frame, None)

            if des_frame is None:
                cv2.imshow('Camera', frame)

            for model_n, (kp_model, des_model, model_img) in self.model_data.items():
                
                global model_name

                if model_n == model_name + '.jpg':
                
                    matches = self.bf.match(des_model, des_frame)
                    matches = sorted(matches, key=lambda x: x.distance)
                    good_matches = [m for m in matches if m.distance < ERROR_DIST]

                    if len(good_matches):
                        max_matches = len(good_matches)
                        
                        new_target = model_name
                        good_matches_for_homography = good_matches

            if max_matches > 0 and new_target:
                if new_target != self.current_target:
                    self.current_target = new_target + '.jpg'
                    kp_model, des_model, model_img = self.model_data[self.current_target]

                src_pts = np.float32([kp_model[m.queryIdx].pt for m in good_matches_for_homography]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches_for_homography]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                '''if self.reavaliate == True:
                    self.vddM, mask2 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    self.reavaliate = False
                    print('Changed vddM')
                
                if self.vddMmuda == 0:
                    print("*"*100)
                    self.vddMmuda = 1
                    self.vddM, mask2 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    print('Changed vddM')
                    print(self.vddM)'''
                #print(f"Matches: {len(good_matches)}")

                if M is not None:
                    try:
                        
                        mainObj = None

                        match(self.current_target):
                            case 'NaCl.jpg':
                                mainObj = self.objNaCl
                            case 'FePt.jpg':
                                mainObj = self.objFePt
                            case 'H2O.jpg':
                                mainObj = self.objH2O
                            case _:
                                print("Modelo não reconhecido")


                        print(f"Model: {self.current_target}, Matches: {len(good_matches)}")
                        #sleep(1)
                        try:
                            projection = projection_matrix(self.camera_parameters, self.vddM)
                            frame = render(frame, mainObj, self.vddM)
                            print(f'Projection with vddM: {projection}')
                        except:
                            projection = projection_matrix(self.camera_parameters, M)
                            frame = render(frame, mainObj, projection)
                            print(f'Projection with M: {projection}')
                    except Exception as e:
                        print(f"Render error: {e}")

        #cv2.imshow('Camera', frame)
        self.display_frame(frame)
        self.schedule = Clock.schedule_once(self.update, 1.0 / 30.0)
        #self.schedule = Clock.schedule_once(self.update, 2.0 / 30.0)

    def display_frame(self, frame):
        # Convert the frame to texture
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture

    def on_leave(self):
        self.capture.release()
        Clock.unschedule(self.schedule)

    def toggle_reavaliate(self):
        self.reavaliate = True
        print(f"Reavaliate: {self.reavaliate}")

class MainApp(MDApp):
    def build(self):
        
        Window.size = (360, 640)

        Builder.load_file('main.kv')
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(BibliotecaScreen(name='biblioteca'))
        #sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(CameraEspecificaScreen(name='cameraespecifica'))
        sm.add_widget(CameraScreen(name='camera'))
        return sm
    
    def on_back_button_pressed(self):
        # Define o comportamento ao pressionar o botão de voltar
        self.root.current = "main"  # Voltar para a tela principal
        print("Botão de retorno pressionado")


if __name__ == '__main__':
    MainApp().run()