import pandas as pd
import os
import csv

directory = 'Data/'
samples = 20000

def CheckRowAmount(filename):
    file = open(directory+filename)
    file = csv.reader(file)
    value = len(list(file))
    if value > samples:
        return True
    else:
        return False


def ReduceFile(filename):
    df = pd.read_csv(directory+filename)

    #Only use recognized drawings and remove unnecessary columns.
    df = df[df["recognized"]==True]
    df = df.drop(columns=['recognized','key_id', 'timestamp'])

    #Lists that are to short or to long are data that is not usable.
    df = df[(df['drawing'].str.len() > 250) & (df['drawing'].str.len() <  3000)]

    #Only keep a certain amount of rows for each class.
    df = df.sample(n=samples, random_state=1).reset_index(drop=True)

    df.to_csv(directory+filename, index=False)


def Data_Transformation():
    for filename in os.listdir(directory):
        if CheckRowAmount(filename):
            try:
                ReduceFile(filename)
                print(filename)
            except:
                pass
        else:
            os.remove(filename)

Data_Transformation()