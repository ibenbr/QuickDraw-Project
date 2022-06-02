import pandas as pd
import os

directory = 'Data/Complete_Data_Raw/'
newdir = "Data/Complete_Data_Filtered/"
samples = 21000

def ReduceFile(filename):
    df = pd.read_csv(directory+filename)

    #Only use recognized drawings and remove unnecessary columns.
    try:
        df = df[df["recognized"]==True]
        df = df.drop(columns=['recognized','key_id', 'timestamp'])
    except:
        pass
    
    #Lists that are to short or to long are data that is not usable.
    df = df[(df['drawing'].str.len() > 250) & (df['drawing'].str.len() <  3000)]

    if len(df) > samples:
        #Only keep a certain amount of rows for each class.
        df = df.sample(n=samples, random_state=1).reset_index(drop=True)
        df.to_csv(newdir+filename, index=False)
    elif len(df) < samples: 
        os.remove(directory+filename)


def Data_Transformation():
    for filename in os.listdir(directory):
            ReduceFile(filename)
            print(filename)

Data_Transformation()