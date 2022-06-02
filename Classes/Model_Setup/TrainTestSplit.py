from tensorflow import keras
import numpy as np

imheight, imwidth = 64, 64  

def SplitData(train_grand, test_grand, num_classes):
    # memory-friendly alternative to train_test_split
    valfrac = 0.2
    cutpt = int(valfrac * train_grand.shape[0])

    np.random.shuffle(train_grand)
    y_train, X_train = train_grand[cutpt: , 0], train_grand[cutpt: , 1:]
    y_val, X_val = train_grand[0:cutpt, 0], train_grand[0:cutpt, 1:]
    y_test, X_test = test_grand[0: , 0], test_grand[0: , 1:]

    y_train = keras.utils.to_categorical(y_train, num_classes)
    X_train = X_train.reshape(X_train.shape[0], imheight, imwidth, 1)

    y_val = keras.utils.to_categorical(y_val, num_classes)
    X_val = X_val.reshape(X_val.shape[0], imheight, imwidth, 1)

    y_test = keras.utils.to_categorical(y_test, num_classes)
    X_test = X_test.reshape(X_test.shape[0], imheight, imwidth, 1)

    return y_train, X_train, y_val, X_val, y_test, X_test
