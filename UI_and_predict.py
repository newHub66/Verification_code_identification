from captcha.image import ImageCaptcha
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from captcha.image import ImageCaptcha
import cv2
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from PIL import Image, ImageTk
from tensorflow import keras
import psutil
import threading


number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
CHAR_SET = number + alphabet + ALPHABET # 得到的CHAR_SET是一个包含62个元素的大列表
CHAR_SET_LEN = len(CHAR_SET) # 26个小写字母+26个大写字母+10个数字 共62个
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160


class Window:
    def __init__(self, win, ww, wh):
        self.win = win
        self.ww = ww
        self.wh = wh
        self.file_names = [] #存放文件夹中文件名称
        self.next_img = 0 # 验证码索引
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))  # 界面启动时的初始位置
        self.win.title("LYH验证码识别")
        self.src_path = None #文件夹路径
        self.img_src_path = None # 文件路径
        self.n = 0               # 文件数目
        self.x_train = []        # 验证码图片
        self.str = []            # 预测结果（字符串形式）

        self.label_src = Label(self.win, text='原图:', font=('微软雅黑', 13)).place(x=0, y=0)


        self.can_src = Canvas(self.win, width=256, height=256, bg='white', relief='solid', borderwidth=1)  # 原图画布
        self.can_src.place(x=50, y=0)
        
        self.can_lic2 = Canvas(self.win, width=240, height=80, bg='white', relief='solid', borderwidth=1)  
        self.can_lic2.place(x=350, y=85)
    

        self.button1 = Button(self.win, text='选择文件', width=10, height=1, command=self.load_show_img)  # 选择文件按钮
        self.button1.place(x=360, y=wh - 60)
        self.button2 = Button(self.win, text='识别验证码', width=10, height=1, command=self.collect)  # 识别验证码按钮
        self.button2.place(x=360, y=wh - 120)
        self.button3 = Button(self.win, text='下一张', width=10, height=1, command=self.next_show)  # 下一张按钮
        self.button3.place(x=100, y=wh - 30)
        self.button3 = Button(self.win, text='上一张', width=10, height=1, command=self.prev_show)  # 上一张按钮
        self.button3.place(x=200, y=wh - 30)
        print('正在启动中,请稍等...')
        print("已启动,开始识别吧！")
        
    # 寻找文件路径，并将第一张图片展示在self.can_src上
    def load_show_img(self):
        self.clear()
        self.next_img = 0
        folder_path = askdirectory()
        if folder_path:
            self.src_path = folder_path
            self.file_names = None
            self.file_names = os.listdir(self.src_path)
            #self.n = len(self.file_names)
            if not self.file_names:
                tkinter.messagebox.showerror('Error', 'No image files found in the selected folder.')
                return
            if self.file_names[0].endswith('.DS_Store'):
                self.file_names.pop(0)
            self.img_src_path = os.path.join(self.src_path, self.file_names[0])
            img_open = Image.open(self.img_src_path)
            if img_open.size[0] * img_open.size[1] > 160 * 60:
                img_open = img_open.resize((320, 120), Image.ANTIALIAS)
            self.img_Tk = ImageTk.PhotoImage(img_open)
            self.can_src.create_image(100, 120, image=self.img_Tk, anchor='center')
            
    # 清屏
    def clear(self):
        self.can_src.delete('all')
        self.can_lic2.delete('all')
        self.img_src_path = None
    
    # 识别验证码
    def collect(self):
        SAVE_PATH = f"tensorflow_demo/deep_learning/"
        model = tf.keras.models.load_model(SAVE_PATH + 'model') 
        self.n = len(self.file_names)
        self.x_train = np.zeros([self.n, 60, 160, 1])
        for i in range(self.n):
            img_src_path = os.path.join(self.src_path, self.file_names[i])
            #print("img_src_path = ",img_src_path)
            img_open = Image.open(img_src_path).convert('L')
            captcha_image = np.array(img_open) # 将图片用数组像素表示
            
            image = tf.reshape(captcha_image, (60, 160, 1))
            self.x_train[i, :] = image
            
        self.display()
        
    # 调用模型，存放预测结果
    def display(self):
        SAVE_PATH = f"tensorflow_demo/deep_learning/"
        model = tf.keras.models.load_model(SAVE_PATH + 'model') 
        
        prediction = model.predict(self.x_train)
        
        max_labels = np.argmax(prediction, axis=2)

        print(prediction.shape)
        print(max_labels)
        
        for k in range(self.n):
            str1 = CHAR_SET[max_labels[k][0]]
            str2 = CHAR_SET[max_labels[k][1]]
            str3 = CHAR_SET[max_labels[k][2]]
            str4 = CHAR_SET[max_labels[k][3]]
            str = str1 + str2 + str3 + str4
            self.str.append(str)
        print(CHAR_SET)
        
        self.display2()
            
    # 展示预测结果
    def display2(self):
        self.img_src_path = os.path.join(self.src_path, self.file_names[self.next_img])
        img_open = Image.open(self.img_src_path)
        if img_open.size[0] * img_open.size[1] > 160 * 60:
            img_open = img_open.resize((320, 120), Image.ANTIALIAS)
        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.can_src.create_image(100, 120, image=self.img_Tk, anchor='center')
        self.can_lic2.create_text(20, 20, text= self.str[self.next_img], anchor='nw', font=('黑体', 35))
        
    #下一张
    def next_show(self):
        if self.next_img < len(self.file_names):
            self.clear()
            self.next_img = self.next_img + 1
        if self.next_img == len(self.file_names):
            self.can_src.create_text(20, 120, text='没有下一张了', anchor='nw', font=('黑体', 28))
            return
        print(self.next_img)
        
        self.display2()
        
    #上一张
    def prev_show(self):
        if -1 < self.next_img <= len(self.file_names):
            self.clear()
            self.next_img = self.next_img - 1
        if self.next_img == -1:
            self.can_src.create_text(20, 120, text='没有上一张了', anchor='nw', font=('黑体', 28))
            return
        print(self.next_img)
               
        self.display2()


if __name__ == '__main__':
    win = Tk()
    ww = 650  
    wh = 300  
    Window(win, ww, wh)
    win.mainloop()
