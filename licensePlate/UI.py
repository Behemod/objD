# -*- coding:utf-8 -*-
# original author: DuanshengLiu
import cv2
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tensorflow import keras
from core import locate_and_correct
from Unet import unet_predict
from CNN import cnn_predict
import sys

class Window:
    def __init__(self, win, ww, wh):
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))  # The initial position when the interface starts
        self.win.title("License plate location, correction and recognition software")
        self.img_src_path = None

        self.label_src = Label(self.win, text='original image:', font=('微软雅黑', 13)).place(x=0, y=0)
        self.label_lic1 = Label(self.win, text='License plate area 1:', font=('微软雅黑', 13)).place(x=700, y=0)
        self.label_pred1 = Label(self.win, text='Recognition result 1:', font=('微软雅黑', 13)).place(x=700, y=85)
        self.label_lic2 = Label(self.win, text='License plate area 2:', font=('微软雅黑', 13)).place(x=700, y=180)
        self.label_pred2 = Label(self.win, text='Recognition result 2:', font=('微软雅黑', 13)).place(x=700, y=265)
        self.label_lic3 = Label(self.win, text='License plate area 3:', font=('微软雅黑', 13)).place(x=700, y=360)
        self.label_pred3 = Label(self.win, text='Recognition result 3:', font=('微软雅黑', 13)).place(x=700, y=445)

        self.can_src = Canvas(self.win, width=512, height=512, bg='white', relief='solid', borderwidth=1)  # original canvas
        self.can_src.place(x=130, y=0)
        self.can_lic1 = Canvas(self.win, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # License plate area 1 canvas
        self.can_lic1.place(x=880, y=0)
        self.can_pred1 = Canvas(self.win, width=245, height=65, bg='white', relief='solid', borderwidth=1)  # License plate recognition 1 canvas
        self.can_pred1.place(x=880, y=90)
        self.can_lic2 = Canvas(self.win, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # License Plate Area 2 Canvas
        self.can_lic2.place(x=880, y=175)
        self.can_pred2 = Canvas(self.win, width=245, height=65, bg='white', relief='solid', borderwidth=1)  # License plate recognition 2 canvas
        self.can_pred2.place(x=880, y=265)
        self.can_lic3 = Canvas(self.win, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # License plate area 3 canvas
        self.can_lic3.place(x=880, y=350)
        self.can_pred3 = Canvas(self.win, width=245, height=65, bg='white', relief='solid', borderwidth=1)  # License plate recognition 3 canvas
        self.can_pred3.place(x=880, y=440)

        self.button1 = Button(self.win, text='Select a document', width=12, height=1, command=self.load_show_img)  # select file button
        self.button1.place(x=680, y=wh - 30)
        self.button2 = Button(self.win, text='Recognize license plate', width=16, height=1, command=self.display)  # Recognition license plate button
        self.button2.place(x=820, y=wh - 30)
        self.button3 = Button(self.win, text='clear all', width=6, height=1, command=self.clear)  # clear all buttons
        self.button3.place(x=1000, y=wh - 30)
        self.unet = keras.models.load_model('licensePlate/unet.h5')
        self.cnn = keras.models.load_model('licensePlate/cnn.h5')
        print('Starting up, please wait...')
        cnn_predict(self.cnn, [np.zeros((80, 240, 3))])
        print("Started, start to identify it!")


    def load_show_img(self):
        self.clear()
        sv = StringVar()
        sv.set(askopenfilename())
        self.img_src_path = Entry(self.win, state='readonly', text=sv).get()  # Get the opened picture
        img_open = Image.open(self.img_src_path)
        if img_open.size[0] * img_open.size[1] > 240 * 80:
            img_open = img_open.resize((512, 512), Image.ANTIALIAS)
        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.can_src.create_image(258, 258, image=self.img_Tk, anchor='center')

    def display(self):
        if self.img_src_path == None:  # Prediction without selecting an image
            self.can_pred1.create_text(32, 15, text='Please select a picture', anchor='nw', font=('黑体', 28))
        else:
            img_src = cv2.imdecode(np.fromfile(self.img_src_path, dtype=np.uint8), -1)  # When reading from a Chinese path, use
            h, w = img_src.shape[0], img_src.shape[1]
            if h * w <= 240 * 80 and 2 <= w / h <= 5:  # Satisfying this condition means that the entire picture may be a license plate, which can be directly recognized without positioning
                lic = cv2.resize(img_src, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # Directly resize to (240,80)
                img_src_copy, Lic_img = img_src, [lic]
            else:  # Otherwise, it is necessary to predict the original image of img_src through unet, get img_mask, realize the license plate location, and then recognize it
                img_src, img_mask = unet_predict(self.unet, self.img_src_path)
                img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)  # Use the locate_and_correct function in core.py for license plate location and correction

            Lic_pred = cnn_predict(self.cnn, Lic_img)  # Using cnn to recognize and predict the license plate, Lic_pred stores the ancestor (license plate picture, recognition result)
            if Lic_pred:
                img = Image.fromarray(img_src_copy[:, :, ::-1])  # img_src_copy[:, :, ::-1] converts BGR to RGB
                self.img_Tk = ImageTk.PhotoImage(img)
                self.can_src.delete('all')  # Before displaying, clear the artboard first
                self.can_src.create_image(258, 258, image=self.img_Tk,
                                          anchor='center')  # The outline of the license plate is drawn on img_src_copy and displayed on the drawing board
                for i, lic_pred in enumerate(Lic_pred):
                    if i == 0:
                        self.lic_Tk1 = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                        self.can_lic1.create_image(5, 5, image=self.lic_Tk1, anchor='nw')
                        self.can_pred1.create_text(35, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))
                    elif i == 1:
                        self.lic_Tk2 = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                        self.can_lic2.create_image(5, 5, image=self.lic_Tk2, anchor='nw')
                        self.can_pred2.create_text(40, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))
                    elif i == 2:
                        self.lic_Tk3 = ImageTk.PhotoImage(Image.fromarray(lic_pred[0][:, :, ::-1]))
                        self.can_lic3.create_image(5, 5, image=self.lic_Tk3, anchor='nw')
                        self.can_pred3.create_text(40, 15, text=lic_pred[1], anchor='nw', font=('黑体', 28))

            else:  # Lic_pred is empty, indicating that it has not been recognized
                self.can_pred1.create_text(47, 15, text='failed to recognize', anchor='nw', font=('黑体', 27))

    def clear(self):
        self.can_src.delete('all')
        self.can_lic1.delete('all')
        self.can_lic2.delete('all')
        self.can_lic3.delete('all')
        self.can_pred1.delete('all')
        self.can_pred2.delete('all')
        self.can_pred3.delete('all')
        self.img_src_path = None

    def closeEvent():  # Clear session() before closing to prevent 'NoneType' object is not callable
        keras.backend.clear_session()
        sys.exit()


if __name__ == '__main__':
    win = Tk()
    ww = 1000  # Window width setting 1000
    wh = 600  # Window height setting 600
    Window(win, ww, wh)
    win.protocol("WM_DELETE_WINDOW", Window.closeEvent)
    win.mainloop()
