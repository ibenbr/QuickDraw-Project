import os
from os import path
import shutil
import pandas as pd

def Create_Testset(directory, classes):
    destination = 'Data/'+str(classes)+'_Categories/testset' 
    exists = os.path.exists(destination)
    
    if not exists:
        os.makedirs(destination)

    for filename in os.listdir(directory):
        df = pd.read_csv(directory+filename)
        df2 = df.tail(100)

        row_count, dummy = df.shape
        if row_count > 20000:
            df.drop(df.tail(100).index,inplace = True)
            df.to_csv(directory+filename, index=False)

        df2.to_csv("Data/"+str(classes)+"_categories/testset/"+filename, mode='w', index=False)

def splitData(amount):
    source = 'Data/Complete_Data_Filtered/'              
    destination = 'Data/'+str(amount)+'_Categories/trainset/' 
    exists = os.path.exists(destination)
    
    if not exists:
        os.makedirs(destination)

    folders = os.listdir(source)               

    for folder in folders[:amount]:                                           
        file = folder                  
        curr_file = source + '\\' + file    
        shutil.copy(curr_file, destination) 
    
    directory = "Data/"+str(amount)+"_Categories/trainset/"
    Create_Testset(directory, amount)