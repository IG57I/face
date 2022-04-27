import tkinter as tk
from tkinter import ttk
# import 
import PIL
from PIL import Image, ImageTk
import cv2
import pickle
import tkinter.filedialog as tf

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Window_class(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.geometry('800x600')
        self.title('Окно классификации')
        self.bind('<Escape>', lambda e: self.destroy())
        self.focus_set()

        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.cv2image = None

        # load models for calssification
        # Для детектирования лиц используем каскады Хаара
        self.cascadePath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        # load model and encoder
        with open('encoder.pickle', 'rb') as file:
            self.encoder = pickle.load(file)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)
        self.recognizer.read('face_recognizer.yml')

        width, height = 800, 600
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.lblimage = ttk.Label(self)
        self.lblimage.pack()
        self.lbltext = ttk.Label(self, text="Ты это")
        self.lbltext.pack()

        self.btn1 = ttk.Button(self,
                               text='Начать',
                               command=self.photo)
        self.btn1.pack(expand=True)

        self.btn2 = ttk.Button(self,
                               text='Правильно',
                               command=self.photo_true)

        self.btn3 = ttk.Button(self,
                               text='Еще раз',
                               command=self.photo)

    def photo_true(self):
        self.destroy()

    #     def photo_false(self):
    #         selff.btn2.destroy()
    #         selff.btn3.destroy()
    def photo(self):
        number_predicted, conf = self.recognizer.predict(
            self.cv2image[self.y: self.y + self.h, self.x: self.x + self.w])
        name = self.encoder.inverse_transform([number_predicted])
        self.lbltext.configure(text=str(name) + '-' + str(round(100 - conf)))
        self.btn2.pack(side='left', expand=True)
        self.btn3.pack(side='right', expand=True)
        self.btn1.destroy()

    def camera(self, loop=True):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            # Получаем лица с изображения
            faces = self.faceCascade.detectMultiScale(self.cv2image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            self.x, self.y, self.w, self.h = faces[0]
            #             x, y, w, h = faces[0]
            # рисуем прямоугольник
            #             cv2.rectangle(cv2image,(x,y),(x+w,y+h),(0,55,0),3)
            cv2.rectangle(self.cv2image, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 55, 0), 3)
            img = PIL.Image.fromarray(self.cv2image)
        except:
            img = PIL.Image.fromarray(self.cv2image)

        imgtk = PIL.ImageTk.PhotoImage(image=img)
        self.lblimage.imgtk = imgtk
        self.lblimage.configure(image=imgtk)
        if loop:
            self.lblimage.after(100, self.camera)


class Window_add(Window_class):
    def __init__(self, parent):
        super().__init__(parent)
        self.title('Окно Дабавления')

        self.num = 0
        self.path = None
        self.lbltext.configure(text='Нажмимайте кнопку Дабавить если есть рамка около лица')
        self.lbltext2 = ttk.Label(self, text="Счетчик фото" + str(self.num))
        self.lbltext2.pack()
        self.btn1.configure(text='Дабавить', command=self.add_photo)

    #         self.btn1 = ttk.Button(self,
    #                 text='Дабавить',
    #                 command=self.add_photo)
    #         self.btn1.pack(expand=True)
    def add_photo(self):
        if self.path == None:
            self.path = tf.askdirectory()
        cv2.imwrite(self.path + '/' + str(self.num) + ".png", self.cv2image)
        self.num += 1
        self.lbltext2.configure(text="Счетчик фото" + str(self.num))


class Window_retrain(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.title('Окно переобучения')

        # Для детектирования лиц используем каскады Хаара
        self.cascadePath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)
        # для кодировки меток
        self.encoder = LabelEncoder()

    def get_images_my(self, path):
        images = []
        labels = []
        for folder in os.listdir(path):
            for file in os.listdir(path + '/' + folder):
                image = Image.open(path + '/' + folder + '/' + file).convert('L')
                image = np.array(image, 'uint8')
                image = cv2.resize(image, (400, 400))
                cv2.imshow("", image)
                # Определяем области где есть лица
                faces = self.faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                # Если лицо нашлось добавляем его в список images, а соответствующий ему номер в список labels
                for (x, y, w, h) in faces:
                    images.append(image[y: y + h, x: x + w])
                    labels.append(folder)
                    # В окне показываем изображение
                    cv2.imshow("", image[y: y + h, x: x + w])
                    cv2.waitKey(50)
        labels = self.encoder.fit_transform(labels)
        return images, labels

    def retrain(self):
        # Путь к фотографиям
        # Получаем лица и соответствующие им номера
        path = tf.askdirectory()
        images, labels = self.get_images_my(path)
        cv2.destroyAllWindows()
        self.recognizer.train(images, np.array(labels))

        self.recognizer.save('face_recognizer.yml')

        with open('encoder.pickle', 'wb') as file:
            pickle.dump(self.encoder, file)
        self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry('300x200')
        self.title('Главной окно')
        self.bind('<Escape>', lambda e: self.destroy())
        self.focus_set()

        # place a button on the root window
        ttk.Button(self,
                   text='Классификация',
                   command=self.open_class).pack(expand=True)
        ttk.Button(self,
                   text='Добавить чнловека',
                   command=self.open_add).pack(expand=True)
        ttk.Button(self,
                   text='Переобучение',
                   command=self.open_retrain).pack(expand=True)

    def open_class(self):
        window_class = Window_class(self)
        window_class.camera()
        window_class.mainloop()

    def open_add(self):
        window = Window_add(self)
        window.camera()
        window.mainloop()

    def open_retrain(self):
        window = Window_retrain(self)
        window.retrain()


#         window.mainloop()


if __name__ == "__main__":
    app = App()
    app.mainloop()