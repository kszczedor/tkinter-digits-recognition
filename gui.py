from tkinter import *
from tkinter import messagebox
import PIL
from PIL import ImageGrab, Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class Main:

    def __init__(self, master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 21
        self.cnn_label = StringVar()
        self.dff_label = StringVar()
        self.digit = StringVar()
        self.path = ".model"
        self.pre_path = "models/"

        self.master.frame
        self.drawWidgets()

        self.c.bind('<Button-1>', self.paintDot)
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.bind('<Button-2>', self.predict)
        self.c.bind('<Button-3>', self.clear)
        self.master.protocol("WM_DELETE_WINDOW", self.close)

        # load models or create new ones
        try:
            self.cnn_model = tf.keras.models.load_model(
                self.pre_path+'cnn' + self.path
                )
        except:
            if messagebox.askyesno(
                "Load model error",
                "Do you want to create new CNN network model?"
            ):
                from network import Network
                nn = Network()
                nn.createCnnModel()
                self.cnn_model = nn.cnnModel
            else:
                self.master.destroy()

        try:
            self.dff_model = tf.keras.models.load_model(
                self.pre_path+'dff' + self.path)
        except:
            if messagebox.askyesno(
                "Load model error",
                "Do you want to create new DFF network model?"
            ):
                from network import Network
                nn = Network()
                nn.createDffModel()
                self.dff_model = nn.dffModel
            else:
                self.master.destroy()

    def paint(self, e):
        if self.old_x and self.old_y:
            self.c.create_line(
                self.old_x, self.old_y, e.x, e.y,
                width=self.penwidth, fill=self.color_fg,
                capstyle=ROUND, smooth=True
            )

        self.old_x = e.x
        self.old_y = e.y

    def paintDot(self, e):
        self.c.create_oval(
            e.x - self.penwidth/2, e.y - self.penwidth/2,
            e.x + self.penwidth/2, e.y + self.penwidth/2, fill=self.color_fg
        )

    def reset(self, e):    
        self.old_x = None
        self.old_y = None

    def clear(self, e=None):
        self.c.delete(ALL)

    def drawWidgets(self):
        # canvas
        self.c = Canvas(self.master, width=280, height=280,  bg=self.color_bg)
        self.c.grid(row=0, column=0, columnspan=28, rowspan=28,
                    sticky=W+E+N+S, padx=5, pady=5)

        # prediction labels
        Label(self.master, text="Prediction").grid(row=0, column=29, 
                                                   sticky=W, columnspan=30)

        Label(self.master, text="CNN:").grid(row=1, column=29 ,sticky=W)
        Label(self.master, text="0", width=3, textvariable=self.cnn_label).grid(row=1, column=30 ,sticky=W)

        Label(self.master, text="DFF:").grid(row=2, column=29 ,sticky=W)
        Label(self.master, text="0", width=3, textvariable=self.dff_label).grid(row=2, column=30 ,sticky=W)

        # predict and clear buttons
        Button(self.master, text="Predict", command=self.predict).grid(row=12, column=29, columnspan=30)
        Button(self.master, text="Clear", command=self.clear).grid(row=13, column=29, columnspan=30)

        # learning section
        Label(self.master, text="Digit:", width=3).grid(row=20, column=29 ,sticky=W)
        Spinbox(self.master, from_=0, to=9, textvariable=self.digit, width=2).grid(row=20, column=30 ,sticky=W)
        Button(self.master, text="Learn", command=self.learn).grid(row=21, column=29, columnspan=30)

    def prepareData(self):
        # image coords
        x = self.master.winfo_rootx() + self.c.winfo_x()
        y = self.master.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()

        # resize to 28x28, anti-aliasing, invertic
        img = PIL.ImageGrab.grab()
        img = img.crop((x, y, x1, y1)).resize((28, 28), resample=Image.LANCZOS)
        img = img.convert('L')
        img = PIL.ImageOps.invert(img)

        # img -> np.array, normalize
        arr = np.array(img)
        arr_n = tf.keras.utils.normalize(arr, axis=0)

        # transform np.array to each model shape
        arr_cnn = np.array([np.expand_dims(arr_n, -1)])
        arr_dff = np.array([[arr_n]])
        return arr_cnn, arr_dff

    def predict(self, e=None):
        arr_cnn, arr_dff = self.prepareData()

        # predict
        prediction = {'cnn': self.cnn_model.predict(arr_cnn),
                      'dff': self.dff_model.predict(arr_dff)}

        # update prediction labels
        self.cnn_label.set(str(np.argmax(prediction['cnn'])))
        self.dff_label.set(str(np.argmax(prediction['dff'])))

    def learn(self):

        y = np.array([int(self.digit.get()[0])])
        x_cnn, x_bp = self.prepareData()

        self.cnn_model.train_on_batch(x_cnn[0:1], y)
        self.dff_model.train_on_batch(x_bp[0:1], y)

    def close(self):
        if messagebox.askyesno(
            "Quit",
            "Do you want to save training progress before quit?"
        ):
            self.cnn_model.save(self.pre_path+'cnn'+self.path)
            self.dff_model.save(self.pre_path+'dff'+self.path)
            self.master.destroy()

        else:
            self.master.destroy()


if __name__ == '__main__':

    root = Tk()
    root.resizable(width=False, height=False)
    Main(root)
    root.mainloop()
