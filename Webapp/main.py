import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow import keras
from rdp import rdp
import numpy as np
from tensorflow import keras
import ast
import numpy as np
from dask import bag
from PIL import Image, ImageDraw 
import os
import csv
import time
imheight, imwidth = 64, 64  
#loading the model
def load_model():
    model = keras.models.load_model('Webapp/model/100_Categories_Model.h5')
    return model

st.write("Don't draw too small, use the canvas as much as possible.")
st.write("The available categories are displayed on your left.")
#transform the data
def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in strokes:
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=0, width=5)
    image = image.resize((imheight, imwidth))
    return np.array(image)/255.

#transform the data
def testArray(drawingXY):
    test_grand = []
    list = {'drawing':[drawingXY]}
    test = pd.DataFrame(list)
    imagebag = bag.from_sequence(test.drawing.values).map(draw_it) 
    testarray = np.array(imagebag.compute())  # PARALLELIZE
    testarray = np.reshape(testarray, (1, -1))    
    labelarray = np.full((test.shape[0], 1),1)
    testarray = np.concatenate((labelarray, testarray), axis=1)
    test_grand.append(testarray)
            
    test_grand = np.array(test_grand.pop()).reshape((-1, (imheight*imwidth+1)))

    del testarray
    del test
    return test_grand

#transform the data
def SplitData(test_grand, num_classes):
    y_test, X_test = test_grand[0: , 0], test_grand[0: , 1:]

    y_test = keras.utils.to_categorical(y_test, num_classes)
    X_test = X_test.reshape(X_test.shape[0], imheight, imwidth, 1)

    return y_test, X_test

#Convert the given stroke
def ConvertStroke(inputList):
    inputList = inputList[1:-1]
    #Remove the first value of the list inside the list ('q') and remove the last 2 values
    temp = []
    for i in inputList:
        i = i[1:-2]
        temp.append(i)
    temp = temp[::2] 
    temp = rdp(temp, epsilon=2)
    # append all x values in 1 list and all y values in 1 list
    x = []
    y = []
    a = 0
    for i in temp:
        x.insert(a, i[0])
        y.insert(a, i[1])
        a = a+1

    #combine the x and y lists 
    drawing = []
    drawing.append(x)
    drawing.append(y)
    return drawing

# predicting the correct category
def predict(insertedfigure):
    test_grand = testArray(insertedfigure)
    y_test, X_test = SplitData(test_grand, 100)
    model = load_model()
    l=os.listdir('Data/'+str(100)+'_Categories/trainset')
    labels=[x.split('.')[0] for x in l]
    labels=sorted(labels)
    prediction = model.predict(X_test)
    predicted_class = labels[np.argmax(prediction)]
    percentage = round(np.amax(prediction)*100,2)
    return predicted_class, percentage

realtime_update = st.sidebar.checkbox("Update in realtime", True)

l=os.listdir('Data/'+str(100)+'_Categories/trainset')
labels=[x.split('.')[0] for x in l]
with st.sidebar:
    st.write("**Available Categories**",sorted(labels))
# Specify canvas parameters in application
drawing_mode = "freedraw"

if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 1, 3)



# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  
    stroke_color="black",
    background_color="white",
    stroke_width=1,
    update_streamlit=realtime_update,
    height=512,
    width=512,
    drawing_mode="freedraw",
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

if canvas_result.json_data is not None:
    figure = []
    this = canvas_result.json_data['objects']
    for x in range(len(this)):
        temp = this[x]['path']
        newStroke = ConvertStroke(temp)
        newStroke = [[item / 2 for item in subl] for subl in newStroke]
        figure.append(newStroke)

    if len(this) != 0:
        with st.spinner("Hold on, I'm predicting..."):
            predicted_class, probability = predict(figure)
            st.title(str(predicted_class)+" | "+str(probability)+" % certainty.")
            st.success('Done!')
