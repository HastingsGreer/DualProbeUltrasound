import itk

from constants import *
import os
import scipy.linalg
print("whoop")

train_volumes_path = volumes_path + "train/"
test_volumes_path = volumes_path + "test/"
train_names = os.listdir(train_volumes_path)
test_names = os.listdir(test_volumes_path)


train_output_dir="prepped_data/train/"
test_output_dir="prepped_data/test/"

#import itkwidgets

import numpy as np
import matplotlib.pyplot as plt

debug=False

import pickle
def load_image_annotation(name, folder):
    image = itk.imread(folder + name)
    annotations = [
        np.argmax(pickle.load(
            open(
                annotations_path + name[:-5] + "axis" + str(i) + ".pickle",
                "rb"
            )
        )[:, :, 3], 0)
        for i in [0, 1]
    ]
    for i in range(2):
        annotations[i] = image.GetSpacing()[i] * annotations[i]
    return image, annotations
class TwoSensorProbe:
    def __init__(self, side_offset, angle):
        self.side_offset = side_offset
        self.angle = angle
    
    def get_origin_direction(self, origin, direction, idx):
        """idx = 1 or -1"""
        origin = np.array(origin) +  idx * self.side_offset * direction[:, 2]
        angle = self.angle * idx
        direction = np.dot(direction, np.array([[1, 0, 0],
                                                [0, np.cos(angle), np.sin(angle)],
                                                [0, -np.sin(angle), np.cos(angle)]]))
        return origin, direction
    
def slice_multiprobe(image, probe, origin, direction):
    res = []
    for idx in [-1, 1]:
        o, d = probe.get_origin_direction(origin, direction, idx)
        if debug:
            pass#print(o, d)
        p = SliceParams(
            o, d,
            width=100,  #mm
            height=100) #mm

        x = spine_slice(image, p)

        x = itk.GetArrayFromImage(x)
        x = x[0]
        res.append(x)

    return res
         

class SliceParams:
    def __init__(self, origin, direction=np.identity(3), width=100, height=100, px=128):
        self.direction=direction
        self.origin=origin
        self.width=width
        self.height=height
        self.px=px
        
        assert len(self.origin) == 3
        
def spine_slice(image, params):
    """
    image is the ITK image to slice from
    params is a SliceParams object"""
    output_spacing = [params.height / params.px, params.width / params.px, 1]
    
    
    output_origin = [
            val + params.direction[j, 0] * -params.height / 2 + params.direction[j, 1] * -params.width / 2
            for j, val in enumerate(params.origin)
        ]
    
    #print(output_origin)
    #print(output_spacing)

    return itk.resample_image_filter(
        image, 
        output_origin = output_origin,
        output_spacing = output_spacing, 
        OutputDirection=itk.matrix_from_array(np.ascontiguousarray(params.direction)), 
        size=[params.px, params.px, 1])
import matplotlib.pyplot as plt


def random_small_rotation(factor = 10):
    R = scipy.linalg.orth(np.random.randn(3, 3).astype(np.float64))
    lam, J = scipy.linalg.eig(R)
    RR = np.real(np.dot(np.dot(J, np.diag(lam)**(1 / factor)), np.linalg.inv(J)))
    return RR

def shitty_ultrasound(y):
    

    y = y + 1000
    y = abs(1 - y[:, 1:] / y[:, :-1])

    denominator = np.cumsum(y, axis=1) + .01
    return np.maximum(-10, np.log(np.abs(y / 4**denominator + .0001 * np.random.randn(128, 128)) ))

def generate_sample(image, annotations, evil_debug=False):
    slice_idx = np.random.randint(50, len(annotations[0]) - 50)

    direction = np.dot(random_small_rotation(), np.array([[0, 0, 1],
                  [-1, 0, 0],
                  [0, -1, 0]], dtype=np.float64))

    origin = np.array([annotations[1][slice_idx], -annotations[0][slice_idx], -slice_idx]) + np.random.randn(3) * 15

    probe = TwoSensorProbe(20, -np.pi /4)


    a, b = slice_multiprobe(image, probe, origin, direction)
    if debug:
        plt.imshow(a)
        plt.show()
        plt.imshow(b)
        plt.show()


    movement_relative_to_image_1 = np.random.randn(3) * 10
    rotation_relative_to_image_1 = random_small_rotation(factor=25)

    direction_2 = np.dot(direction, rotation_relative_to_image_1)
    origin_2 = origin + np.transpose(np.dot(direction, movement_relative_to_image_1.transpose()))

    c, d = slice_multiprobe(image, probe, origin_2, direction_2)
    if debug: 
        plt.imshow(c)
        plt.show()
        plt.imshow(d)
        plt.show()
    
    horror = {}
    if evil_debug:
        horror = {k:v for k, v in locals().items() if not k.startswith('__')}
        
    return {"data": [a, b, c, d], "classes": [movement_relative_to_image_1, rotation_relative_to_image_1], "locals":horror}


def generate_data(volume_path, names):
    res = []
    for name in names:
        #print(name)

        image, annotations = load_image_annotation(name, volume_path)

        

        for _ in range(128):
            res.append(generate_sample(image, annotations))

    return res

import os
def load_dataset(path, simulate_ultrasound=False):
    dataset = generate_data(path, os.listdir(path))

    data = []
    classes = []
    for elem in dataset:
        if not simulate_ultrasound:
            data_entry = np.array(np.stack(elem["data"], axis=-1), dtype=np.float32)
            data_entry += 1000
            data_entry /= 2000
        else: 
            data_entry = [shitty_ultrasound(d) for d in elem["data"]]
            data_entry = np.stack(data_entry, axis=-1) / 10 + 1
            data_entry = np.nan_to_num(data_entry)
            data_entry = np.clip(data_entry, -5, 5)

        
        data.append(data_entry)

        class_entry = elem["classes"]
        class_entry = np.concatenate([class_entry[0] / 4, class_entry[1].flatten() * 40])
        class_entry[[3, 7, 11]] -= 40
        classes.append(class_entry)
    data = np.array(data)
    classes = np.array(classes)
    return data, classes