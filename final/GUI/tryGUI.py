from tkinter import *
from PIL import ImageTk, Image
import cv2
import time
import pandas as pd
from camera import Camera


count = 0
images_prefix = "images/"
imagenames = [images_prefix+"img7.jpg",images_prefix+"img81.jpg",images_prefix+"img203.jpg"]
images = []
df = pd.read_csv("images/tagged_data.csv")

Cam = Camera(model_path="../100_basic_SVC1.model")

class GUIDemo(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.grid()
        self.createWidgets()
 
    def createWidgets(self):
        self.numberText = Label(self)
        self.numberText["text"] = "Number:"
        self.numberText.grid(row=0, column=1)
        self.numberField = Entry(self)
        self.numberField["width"] = 50
        self.numberField.grid(row=0, column=2, columnspan=6)
 
        self.nameText = Label(self)
        self.nameText["text"] = "Name:"
        self.nameText.grid(row=1, column=1)
        self.nameField = Entry(self)
        self.nameField["width"] = 50
        self.nameField.grid(row=1, column=2, columnspan=6)
         
        # self.new = Button(self)
        # self.new["text"] = "New"
        # self.new.grid(row=2, column=0)
        # self.new["command"] =  self.newMethod
        # self.load = Button(self)
        # self.load["text"] = "Load"
        # self.load.grid(row=2, column=1)
        # self.load["command"] =  self.loadMethod
        self.save = Button(self)
        self.save["text"] = "Save"
        self.save.grid(row=2, column=1)
        self.save["command"] =  self.saveMethod
        # self.encode = Button(self)
        # self.encode["text"] = "Encode"
        # self.encode.grid(row=2, column=3)
        # self.encode["command"] =  self.encodeMethod
        # self.decode = Button(self)
        # self.decode["text"] = "Decode"
        # self.decode.grid(row=2, column=4)
        # self.decode["command"] =  self.decodeMethod
        self.clear = Button(self)
        self.clear["text"] = "Clear"
        self.clear.grid(row=2, column=2)
        self.clear["command"] =  self.clearMethod
        self.delete = Button(self)
        self.delete["text"] = "Delete"
        self.delete.grid(row=2, column=3)
        self.delete["command"] =  self.deleteMethod
        # self.copy = Button(self)
        # self.copy["text"] = "Copy"
        # self.copy.grid(row=2, column=6)
        # self.copy["command"] =  self.copyMethod
 
        self.displayText = Label(self)
        self.displayText["text"] = "something happened"
        self.displayText.grid(row=3, column=2, columnspan=8)

        # camera
        self.camera = Button(self)
        self.camera["text"] = "Camera"
        self.camera.grid(row=2, column=4)
        self.camera["command"] =  self.cameraMethod

        # try image
        path = "images/img7.jpg"
        p = cv2.imread(path)
        # notice!! cv2.imread = BGR, not RGB
        p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        #img = ImageTk.PhotoImage(Image.open(path))
        img = ImageTk.PhotoImage(Image.fromarray(p, 'RGB'))
        self.picture = Label(self, image=img)
        self.picture.image = img
        #self.image = Canvas(self)
        # panel.grid(row=3, column=0, columnspan=2, rowspan=2,
        #         sticky=W+E+N+S, padx=5, pady=5)
        self.picture.grid(row=0, column=0, rowspan=6, sticky=NW)
        #self.image.create_image(0,0, image=img)
        #panel.pack(side="bottom", fill="both", expand="yes")

     
    def newMethod(self):
        self.displayText["text"] = "This is New button."
        
 
    def loadMethod(self):
        self.displayText["text"] = "This is Load button."
 
    def saveMethod(self):
        self.displayText["text"] = "This is Save button."
 
    def encodeMethod(self):
        self.displayText["text"] = "This is Encode button."
 
    def decodeMethod(self):
       self.displayText["text"] = "This is Decode button."
 
    def clearMethod(self):
       self.displayText["text"] = "This is Clear button."
 
    def deleteMethod(self):
       self.displayText["text"] = "This is Delete button."

    def cameraMethod(self):
       self.displayText["text"] = "This is Camera button."
       Cam.start()
       images = [p for p in Cam.images if p not in images]

    def redraw(self, newimage):
        p = cv2.cvtColor(newimage, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(p, 'RGB'))
        self.picture = Label(self, image=img)
        self.picture.image = img
        self.picture.grid(row=0, column=0, rowspan=6, sticky=NW)
        


class Application(Frame):
    def say_hi(self):
        print "hi there, everyone!"

    def createWidgets(self):
        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit

        self.QUIT.pack({"side": "top"})

        self.hi_there = Button(self)
        self.hi_there["text"] = "Hello",
        self.hi_there["command"] = self.say_hi

        self.hi_there.pack({"side": "left"})

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()


if __name__ == "__main__":


    count = 0
    
    root = Tk()
    app = GUIDemo(master=root)
    root.update_idletasks()
    root.update()

    # app.mainloop()

    while True:
        if len(images) != 0:
            for p in images:
                app.redraw(p)
                root.update_idletasks()
                root.update()
                time.sleep(10)
            images = []
        root.update_idletasks()
        root.update()


    # import Tkinter

    # root = Tkinter.Tk()
    # canvas = Tkinter.Canvas(root)
    # canvas.grid(row = 0, column = 0)
    # photo = Tkinter.PhotoImage(master=root, file = 'Aaron.gif')
    # image1 = canvas.create_image(0,0, image=photo)
    # root.mainloop()




