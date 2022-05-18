import os
import shutil

def split(amount):
    source = 'Data/Complete_Data/'              
    destination = 'Data/'+str(amount)+'_Categories/' 
    exists = os.path.exists(destination)

    if not exists:
        os.makedirs(destination)
        
    folders = os.listdir(source)               

    for folder in folders[:amount]:                                           
        file = folder                  
        curr_file = source + '\\' + file    
        shutil.copy(curr_file, destination) 