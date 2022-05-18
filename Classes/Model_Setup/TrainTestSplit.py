from tensorflow import keras
import numpy as np

imheight, imwidth = 32, 32  

def SplitData(train_grand, num_classes):
    # memory-friendly alternative to train_test_split?
    valfrac = 0.3
    cutpt = int(valfrac * train_grand.shape[0])

    np.random.shuffle(train_grand)
    y_train, X_train = train_grand[cutpt: , 0], train_grand[cutpt: , 1:]
    y_val, X_val = train_grand[0:cutpt, 0], train_grand[0:cutpt, 1:] #validation set is recognized==True

    #del train_grand

    y_train = keras.utils.to_categorical(y_train, num_classes)
    X_train = X_train.reshape(X_train.shape[0], imheight, imwidth, 1)
    y_valid = keras.utils.to_categorical(y_val, num_classes)
    X_valid = X_val.reshape(X_val.shape[0], imheight, imwidth, 1)

    y_val = y_valid[:len(y_valid)//2]
    X_val = X_valid[:len(X_valid)//2]

    y_test = y_valid[len(y_valid)//2:]
    X_test = X_valid[len(X_valid)//2:]

    return y_train, X_train, y_val, X_val, y_test, X_test