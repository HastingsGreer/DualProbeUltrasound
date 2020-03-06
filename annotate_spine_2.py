
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

import cv2
import constants

path = constants.volumes_path


folder = path
outfolder = constants.annotations_path

class ManualSegmenter:
    def __init__(self, image):

        self.image = image

        self.mask = image * 0

        self.line_idx = 0

        self.line = [(0, 0), (0, 0)]
        self.interactive_align()


    def interactive_align(self):
        self.fig = plt.figure()

        self.ax = self.fig.add_subplot(111)
        self.im_disp_obj = self.ax.imshow(self.image)
        self.mask_disp_obj = self.ax.imshow(self.mask)
        

        self.ax.set_title('click on points')
        
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
       
        plt.show()
   

    def draw(self):
        self.im_disp_obj.set_data(self.image)
        self.mask_disp_obj.set_data(self.mask)
        self.fig.canvas.draw()
    def on_click(self, event):
        print(event)
        self.line[self.line_idx] = int(event.xdata), int(event.ydata)
        if self.line_idx == 0:
            
            self.line_idx = 1;
        else:
            self.oldmask = self.mask.copy()
            cv2.line(self.mask,self.line[0],self.line[1],(255,0,0, 255),1)
            self.draw()
            
            self.line_idx = 0;

       
        

    def on_key_press(self, event):
        if event.key == 'a':
            self.image = self.image * 2
            self.draw()
            

        

        if event.key == 'u':
            self.mask = self.oldmask
            self.draw()
            self.line_idx = 0
        
        if event.key == 'd':
            self.gridOnScreen[:, self.activeIndex] = 0, 0
            self.process_updated_annotation()
import random
import nrrd
already_segmented = set(os.listdir(outfolder))
candidates = os.listdir(folder)
random.shuffle(candidates)
import pickle

def image_to_3channel(arr):
    arr = np.expand_dims(arr, -1)
    arr = np.repeat(arr, 4, -1)
    arr = arr - np.min(arr)
    arr =  arr / np.max(arr)
    arr[:, :, 3] = 1
    print(arr.shape)
    return arr
for fname in candidates:
    print(fname)
    if fname[:-5] + "axis0.pickle" not in already_segmented:
        full_name = os.path.join(folder, fname)
        print(full_name)

        if full_name[-5:] in ['.nrrd']:
            t2 = nrrd.read(full_name)[0]
            
            for axis in [0, 1]:
                m = ManualSegmenter(image_to_3channel(
                    np.max(t2, axis)))
                pickle.dump(m.mask, open(outfolder + "/" 
                    +
                    fname[:-5] + "axis" + str(axis)+ ".pickle", "wb"))
            
                