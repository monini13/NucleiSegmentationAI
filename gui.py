import tkinter as tk 
from tkinter import *
from tkinter import filedialog
import os
from PIL import Image, ImageTk
from predict import predict, get_actual_mask 
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

class Window(Frame):
    def __init__(self, weights_path, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pos = []
        self.master.title("Nuclei Segmentation")
        self.pack(fill=BOTH, expand=1)
        menu = Menu(self.master)
        self.master.config(menu=menu)
        file = Menu(menu)
        file.add_command(label="Select Image", command=self.uploadImage)
        file.add_command(label="Predict", command=self.show_prediction)
        menu.add_cascade(label="File", menu=file)
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image = None
        self.image2 = None
        self.weights_path = weights_path
        
        frm = Frame(self.master)
        frm.pack(side=BOTTOM, padx=15, pady=15)
        btn1 = Button(frm, text="Select Image", command=self.uploadImage)
        btn1.pack(side=tk.LEFT)
        btn2 = Button(frm, text="Predict", command = self.show_prediction)
        btn2.pack(side=tk.LEFT, padx=30)
    
    
    def uploadImage(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if not filename:
            return
        img = Image.open(filename).convert("RGB")
        if not img:
            return
        #feed input into model here, output into ./result.png
        self.predicted_mask = predict(self.weights_path,img)  # PIL Image
        base_name = os.path.basename(filename)
        base_name = os.path.splitext(base_name)[0]
        labels_list = os.listdir('./Test/Labels')
        label = base_name + ".mat"
        label = loadmat('./Test/Labels/'+label)
        true_mask = get_actual_mask(label)
        self.true_mask = Image.fromarray(np.uint8(true_mask*255)).convert('RGB')
        img = img.resize((400, 400))
        w, h = img.size
        width, height = root.winfo_width(), root.winfo_height()
        self.render = ImageTk.PhotoImage(img)
        if self.image:
            self.canvas.delete(self.text)
        self.image = self.canvas.create_image((w / 3, h / 3), image=self.render)
        self.canvas.move(self.image, 80, 0)   
        self.text = self.canvas.create_text(170,380, fill="black",font="Times 20 bold",
                        text="Input: " + base_name)

        
    def show_prediction(self):
        if not hasattr(self, 'predicted_mask'):
            return
        load = self.predicted_mask
        load = load.resize((400, 400))
        load_true_mask = self.true_mask
        load_true_mask = load_true_mask.resize((400, 400))
        w, h = load.size
        width, height = root.winfo_screenmmwidth(), root.winfo_screenheight()
        self.render2 = ImageTk.PhotoImage(load_true_mask)
        self.image2 = self.canvas.create_image((w / 3, h / 3), image=self.render2)
        self.canvas.move(self.image2, 500, 0)
        self.canvas.create_text(620,380, fill="black",font="Times 20 bold",
                    text="True Mask")
        self.render3 = ImageTk.PhotoImage(load)
        self.image3 = self.canvas.create_image((w / 3, h / 3), image=self.render3)
        self.canvas.move(self.image3, 930, 0)   
        self.canvas.create_text(1050,380, fill="black",font="Times 20 bold",
                    text="Predicted Mask")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Nuclei Segmentation")
    root.geometry('1280x600')
    weights_path = "./weights_3channel_dropout_1"
    app = Window(weights_path,root)
    root.mainloop()
