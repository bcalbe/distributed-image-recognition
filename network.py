import tensorflow as tf 
import tensorflow.python.keras as keras
import tensorflow.keras.layers as layers
#from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import numpy as np
import dataset as Data
import re
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

save_dir = "./model/"


def Get_VGG19():
    feature_extraction = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(32,32,3))
    flatten = tf.keras.layers.Flatten()(feature_extraction.output)
    dense1 = tf.keras.layers.Dense(2048,activation = 'relu')(flatten)
    dense2 = tf.keras.layers.Dense(10,activation = 'softmax')(dense1)
    return tf.keras.Model(inputs = feature_extraction.inputs, outputs = dense2)

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
        if output > 0.7:
            return outputs , i
    return output, i
        
 

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


def get_activation(model,selected_layers,target_data):
    activations = []
    layers = [layer.name for layer in model.layers]
    weights = [model.get_layer(name).weights for name in selected_layers]
    outputs = [model.get_layer(name).output for name in selected_layers]
    feature_extractor = keras.Model(model.inputs,outputs)
    ###可再加个activation selection的功能####

    ########################################
    activations = feature_extractor(target_data)
    print(layers)
    return activations,feature_extractor


def APOZ(activations):
    threshold = 50
    layers_indexs = []
    for activation in activations:
        indexs = []
        
        activation = tf.transpose(activation, perm = [3,0,1,2])
        shape = activation.shape[0]
        for _,features in enumerate(activation):
            size = tf.size(features).numpy()
            num_zeros = tf.size(features[features == 0]).numpy()
            APOZ = 100 * num_zeros/size
            if (APOZ > threshold) & (indexs.count(0) < activation.shape[0]-1):
                indexs.append(0)
            else:
                indexs.append(1)
        layers_indexs.append(indexs)
    return layers_indexs


def replace_layer(model,selected_layers,layer_index,feature_extractor,classes):
# network_structure = ['input','conv','conv','pool','conv',''conv,'pool','flatten','dense']
    layers_name = [layer.name for layer in model.layers]

    #1)先遍历原模型，把conv曾按照apoz结果设置
    new_model = keras.Sequential()
    #for layer in self.model.layers[:len(self.model.layers)-1]:#改为 in network_structure
    for name in layers_name:
        if "input" in name:
            new_model.add(model.get_layer(name))
        elif 'conv' in name:
            index = selected_layers.index(name)
            channels = layer_index[index]
            channels = np.array(channels)
            num_channels = channels.sum()
            new_layer = keras.layers.Conv2D(
                num_channels, kernel_size=(3, 3), activation="relu",strides = (1,1),padding = 'same')
            new_model.add(new_layer) 
        elif 'pool' in name:#改为 pool flatten 和dense分情况设置而不是直接copy元模型
            new_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        elif name.startswith('flatten'):
            new_model.add(layers.Flatten())
        elif name.startswith('dense'):
            new_model.add(layers.Dense(128,activation = "relu",kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            new_model.add(layers.Dense(1,activation = "sigmoid",kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            break
    
    new_conv_layer = [layer.name for layer in new_model.layers if 'conv' in layer.name ]
    print("new_model: {}".format(new_conv_layer)) 
    print("new_model_all: {}".format([layer.name for layer in new_model.layers]))
    new_model.save("./model/noweights/VGG19_{}_noweights.h5".format(classes))
    
    #2)然后再根据apoz结果得到想要的weight，在进行权重初始化
    #得到一层中index为1的layer
    old_weights = [model.get_layer(name).weights for name in selected_layers ]
    for i,name in enumerate(selected_layers):
        ###更新当前layer的权值
        current_layer = new_model.get_layer(new_conv_layer[i])
        current_old_weights = old_weights[i][0]
        bias = old_weights[i][1]
        current_old_weights = tf.transpose(current_old_weights,perm = [3,0,1,2])
        index = np.array(layer_index[i])
        new_weights = current_old_weights[index == 1]
        new_weights = tf.transpose(new_weights,perm = [1,2,3,0])
        bias = bias[index == 1]
        current_layer.weights[0].assign(new_weights)
        current_layer.weights[1].assign(bias)
        ###如果有下一个 conv layer, 也进行剪裁
        if i < len(new_conv_layer)-1:
            next_layer = new_model.get_layer(new_conv_layer[i+1])
            next_weights = tf.transpose(old_weights[i+1][0],perm = [2,0,1,3])
            next_weights = next_weights[index == 1]
            next_weights = tf.transpose(next_weights,perm =[1,2,0,3])
            old_weights[i+1][0] = next_weights
    return new_model

def prune(model,target_data,classes):
        #selected_layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']
        selected_layers = [layer.name for layer in model.layers if "conv" in layer.name ]
        activations,feature_extractor = get_activation(model,selected_layers,target_data)
        layer_index = APOZ(activations)
        prune_model = replace_layer(model,selected_layers,layer_index,feature_extractor,classes)
        prune_model.save("./model/prune/VGG19_{}_prune.h5".format(classes))
        return prune_model
        

def test_layer(model,data):
    outputs = [layer.name for layer in model.layers]
    first_layer = keras.Model(model.inputs,model.get_layer(outputs[1]).output)
    results = first_layer(data)
    return 0






if __name__ == "__main__":
    VGG19 = tf.keras.models.load_model("./model/VGG19_11.h5")
    #VGG19.get_layer("vgg19").summary()
    VGG19.summary()
    print([layer.name for layer in VGG19.layers ])
    Dataset = Data.dataset("cifar10")
    x_train, y_train , x_test, y_test = Dataset.load_data()
    # serial = Get_SerialModel()
    # serial.summary()
    x_retrain_train, y_retrain_train , x_retrain_test, y_retrain_test = Dataset.split_data(x_train, y_train , x_test, y_test,0)
    # test_model = tf.keras.models.load_model("./model/prune/VGG19_0_prune.h5")
    # test_layer(test_model,x_retrain_train)
    for i in range(10):
        pmodel = prune(VGG19,x_train[y_train == i],i)


