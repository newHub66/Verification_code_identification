from captcha.image import ImageCaptcha
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import os

from captcha.image import ImageCaptcha

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

CHAR_SET = number + alphabet + ALPHABET # 得到的CHAR_SET是一个包含62个元素的大列表
CHAR_SET_LEN = len(CHAR_SET) # 26个小写字母+26个大写字母+10个数字 共62个
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
# 该函数的作用是随机生成4位验证码 返回的captcha_text是一个含有四位字母或数字组成的列表
def random_captcha_text(char_set=None, captcha_size=4):
    if char_set is None:
        char_set = number + alphabet + ALPHABET

    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text
def gen_captcha_text_and_image(width=160, height=60, char_set=CHAR_SET):
    image = ImageCaptcha(width=width, height=height)

    captcha_text = random_captcha_text(char_set) # 调用已定义好的random_captcha_text函数 得到了随机的四位验证码的列表
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text) # 调用验证码生成函数，captcha为生成的验证码图片

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image) # 将图片用数组像素表示
    return captcha_text, captcha_image


text, image = gen_captcha_text_and_image(char_set=CHAR_SET)
MAX_CAPTCHA = len(text)
print('CHAR_SET_LEN=', CHAR_SET_LEN, ' MAX_CAPTCHA=', MAX_CAPTCHA)

def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img

def text2vec(text):
    vector = np.zeros([MAX_CAPTCHA, CHAR_SET_LEN]) # max_captcha为4，char_set_len为62
    for i, c in enumerate(text): # enumerate是枚举的含义 i是索引从0到4，c为4个验证码文字
        idx = CHAR_SET.index(c) # idx为4位验证码文字在62个大列表中的索引
        vector[i][idx] = 1.0
    return vector


def vec2text(vec):
    text = []
    for i, c in enumerate(vec):
        text.append(CHAR_SET[c])
    return "".join(text)

x_train = np.zeros([10000, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  # batch_x为生成的验证码图片，通道数为1的灰色图片
y_train = np.zeros([10000, MAX_CAPTCHA, CHAR_SET_LEN])  # batch_y为标签，4*62的矩阵

print("正在生成")
save_folder = "datasets/train"
for i in range(10000):
    text,image = gen_captcha_text_and_image(char_set=CHAR_SET)
    
    # 文件保存，需要可以打开
    #img = Image.fromarray(image)      
    #img.save(os.path.join(save_folder, f"{text}.png"))
    image = tf.reshape(convert2gray(image), (IMAGE_HEIGHT, IMAGE_WIDTH, 1)) # （60，160，1）
    x_train[i, :] = image
    y_train[i, :] = text2vec(text)

print("生成完毕")
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3)))
model.add(tf.keras.layers.PReLU())
model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

model.add(tf.keras.layers.Conv2D(64, (5, 5)))
model.add(tf.keras.layers.PReLU())
model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

model.add(tf.keras.layers.Conv2D(128, (5, 5)))
model.add(tf.keras.layers.PReLU())
model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN)) # Dense全连接的输出为4*62
model.add(tf.keras.layers.Reshape([MAX_CAPTCHA, CHAR_SET_LEN]))
model.add(tf.keras.layers.Softmax())

model.compile(optimizer='Adam',
                metrics=['accuracy'],
                loss='categorical_crossentropy')


history = model.fit(x_train, y_train, batch_size=32, epochs=50)

SAVE_PATH = f"tensorflow_demo/deep_learning/"

try:
    tf.saved_model.save(model, SAVE_PATH + 'model') # 将模型保存到指定路径下
except Exception as e:
    print('#######Exception', e)

print("训练完成")
