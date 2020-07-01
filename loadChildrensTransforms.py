import itk
import os
import constants
import numpy as np

def loadFile(name):
    imageType = itk.Image[itk.F, 3]
    
    readerType = itk.ImageFileReader[imageType]
    
    reader = readerType.New()
    
    reader.SetFileName(name)
    
    im = reader.GetOutput()
    
    
    reader.Update()
    print(im.GetSpacing())
    return np.array(itk.PyBuffer[imageType].GetArrayFromImage(im))
import xml.etree.ElementTree 
def getTransforms(filename):
    
 
    with open(filename, 'rb') as f:
        lines = f.readlines()
    from collections import defaultdict
    fields = defaultdict(lambda: [])
    for entry in [str(l)[11:].split(" ") for l in lines if l[:3] == b'Seq']:
        frame, kind = entry[0].split("_")
        fields[kind].append(entry[1:])



    
    ImageToReference = fields['ImageToReferenceTransform']
    ImageToReference = np.array(ImageToReference)

    ImageToReference = ImageToReference[:, 1:-1].reshape(-1, 4, 4)

    ImageToReference = np.array(ImageToReference, dtype=np.float64)

    return ImageToReference
name = os.path.join(constants.volumes_path, "HultrasoundI2R/h-20180315_120244_ImageToReference.mha")
transforms = getTransforms(name)