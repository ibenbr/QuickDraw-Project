import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from dask import bag
from PIL import Image, ImageDraw 

imheight, imwidth = 64, 64  

def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=0, width=5)
    image = image.resize((imheight, imwidth))
    return np.array(image)/255.

def trainArray(path, num_classes, ims_per_class):
    path = path+"trainset/*.csv"
    train_grand = []
    class_paths = glob(path)
    for i,c in enumerate(tqdm(class_paths[0: num_classes])):
        train = pd.read_csv(c, usecols=['drawing'], nrows=ims_per_class*5//4).head(ims_per_class)
        imagebag = bag.from_sequence(train.drawing.values).map(draw_it) 
        trainarray = np.array(imagebag.compute())  # PARALLELIZE
        trainarray = np.reshape(trainarray, (ims_per_class, -1))    
        labelarray = np.full((train.shape[0], 1), i)
        trainarray = np.concatenate((labelarray, trainarray), axis=1)
        train_grand.append(trainarray)
        
    train_grand = np.array([train_grand.pop() for i in np.arange(num_classes)]) #less memory than np.concatenate
    train_grand = train_grand.reshape((-1, (imheight*imwidth+1)))

    del trainarray
    del train
    return train_grand

def testArray(path, num_classes, ims_per_class):
    path = path+"testset/*.csv"
    test_grand = []
    class_paths = glob(path)
    for i,c in enumerate(tqdm(class_paths[0: num_classes])):
        test = pd.read_csv(c, usecols=['drawing'], nrows=ims_per_class*5//4).head(ims_per_class)
        imagebag = bag.from_sequence(test.drawing.values).map(draw_it) 
        testarray = np.array(imagebag.compute())  # PARALLELIZE
        testarray = np.reshape(testarray, (ims_per_class, -1))    
        labelarray = np.full((test.shape[0], 1), i)
        testarray = np.concatenate((labelarray, testarray), axis=1)
        test_grand.append(testarray)
        
    test_grand = np.array([test_grand.pop() for i in np.arange(num_classes)]) #less memory than np.concatenate
    test_grand = test_grand.reshape((-1, (imheight*imwidth+1)))

    del testarray
    del test
    return test_grand