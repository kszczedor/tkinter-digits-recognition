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
        self.cnn_model = tf.keras.models.load_model('cnn_test.model')
        self.bp_model = tf.keras.models.load_model('dff_test.model')
        self.digit = StringVar()

        self.master.frame
        self.drawWidgets()

        self.c.bind('<Button-1>', self.paintDot)
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.c.bind('<Button-2>', self.predict)
        self.c.bind('<Button-3>', self.clear)
        self.master.protocol("WM_DELETE_WINDOW", self.close)


    def paint(self, e):
        
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, e.x, e.y,
                width=self.penwidth, fill=self.color_fg, 
                capstyle=ROUND, smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def paintDot(self, e):

        self.c.create_oval(e.x - self.penwidth/2, e.y - self.penwidth/2, 
            e.x + self.penwidth/2, e.y + self.penwidth/2, fill=self.color_fg)

    def reset(self, e):
       
        self.old_x = None
        self.old_y = None

    def clear(self, e=None):
        
        self.c.delete(ALL)

    def drawWidgets(self):
        
        # canvas
        self.c = Canvas(self.master, width=280, height=280,  bg=self.color_bg)
        self.c.grid(row=0, column=0, columnspan=28, rowspan=28,sticky=W+E+N+S, padx=5, pady=5)

        # top labels
        Label(self.master, text="CNN:").grid(row=0, column=29 ,sticky=W)
        Label(self.master, text="0", width=3).grid(row=0, column=30 ,sticky=W)
        Label(self.master, text="BP:").grid(row=1, column=29 ,sticky=W)
        Label(self.master, text="0", width=3).grid(row=1, column=30 ,sticky=W)

        # predict, clear buttons
        Button(self.master, text="Predict", command=self.predict).grid(row=12, column=29, columnspan=30)
        Button(self.master, text="Clear", command=self.clear).grid(row=13, column=29, columnspan=30)

        # learning section
        Label(self.master, text="Digit:", width=3).grid(row=20, column=29 ,sticky=W)
        self.spinbox = Spinbox(self.master, from_=0, to=9, textvariable=self.digit, width=2)
        self.spinbox.grid(row=20, column=30 ,sticky=W)
        Button(self.master, text="Learn", command=self.learn).grid(row=21, column=29, columnspan=30)

    def prepareData(self):

        x = self.master.winfo_rootx() + self.c.winfo_x()
        y = self.master.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()

        img = PIL.ImageGrab.grab().crop((x,y,x1,y1)).resize((28, 28), resample=Image.LANCZOS).convert('L')
        img = PIL.ImageOps.invert(img)

        arr = np.array(img)
        arr_n = tf.keras.utils.normalize(arr, axis=0)

        arr_cnn = np.array([np.expand_dims(arr_n,-1)])
        arr_bp = np.array([[arr_n]])
        return arr_cnn, arr_bp


    def predict(self, e=None):

        arr_cnn, arr_bp = self.prepareData()
        prediction = {'cnn':self.cnn_model.predict(arr_cnn), 'bp':self.bp_model.predict(arr_bp)}

        Label(self.master, text=str(np.argmax(prediction['cnn'])), width=3).grid(row=0, column=30 ,sticky=W)
        Label(self.master, text=str(np.argmax(prediction['bp'])), width=3).grid(row=1, column=30 ,sticky=W)


        #plt.imshow(arr, cmap = plt.cm.binary)    
        #plt.title("cnn: {} | bp: {}".format(np.argmax(prediction['cnn']), np.argmax(prediction['bp'])))
        #plt.show()
        #print("cnn: {} | bp: {}".format(np.argmax(prediction['cnn']), np.argmax(prediction['bp'])))
        
    def learn(self):

        y =  np.array([int(self.digit.get()[0])])
        x_cnn, x_bp = self.prepareData()

        self.cnn_model.train_on_batch(x_cnn[0:1], y)
        self.bp_model.train_on_batch(x_bp[0:1], y)

    def close(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.master.destroy()


if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    root = Tk()
    Main(root)
    root.resizable(width=False, height=False)
    root.mainloop()