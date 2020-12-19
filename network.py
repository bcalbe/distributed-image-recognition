import tensorflow as tf 
import tensorflow.python.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import numpy as np


save_dir = "./model/"


def Get_VGG19():
    inputs = tf.keras.layers.Input(shape=[32,32, 3])
    feature_extraction = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(32,32,3))
    x = feature_extraction(inputs)
    flatten = tf.keras.layers.Flatten()(x)
    dense1 = tf.keras.layers.Dense(2048,activation = 'relu')(flatten)
    dense2 = tf.keras.layers.Dense(10,activation = 'softmax')(dense1)
    return tf.keras.Model(inputs = inputs, outputs = dense2)

def Get_SerialModel():
    Sub_models = []
    for i in range(10):
        model_name = "VGG19_{}.h5".format(i)
        model = tf.keras.models.load_model(save_dir+model_name)
        Sub_models.append(model)
    return Sub_models

def run_SerialModel(Sub_models,inputs):
    outputs = []
    for i,model in enumerate(Sub_models):
        output = model.predict(inputs)
        outputs.append(output)
    return outputs
        
 

class Serial_Model(tf.keras.Model):

    def __init__(self):
        super(Serial_Model,self).__init__()
        for i in range(10):
            self.Sub_models = []
            self.is_stop = False
            model_name = "VGG19_{}.h5".format(i)
            model = tf.keras.models.load_model(save_dir+model_name)
            self.Sub_models.append(model)
    @tf.function
    def call(self,inputs):
        for i,model in enumerate(self.Sub_models):
            output = model(inputs)
            if output > 0.7:
                return output,i
        return output,i


if __name__ == "__main__":
    VGG19 = Get_VGG19()
    VGG19.summary()
    serial = Get_SerialModel()
    serial.summary()


