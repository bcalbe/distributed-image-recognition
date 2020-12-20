import tensorflow as tf 
import tensorflow.python.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import numpy as np
import dataset as Data
import network as Net
import os
import time

save_dir = "./model/prune/"

def train(model,train_data,train_label,model_name = "VGG19_2.h5"):

    def scheduler(epoch,lr):
        if epoch <5:
            return lr
        elif epoch <8:
            return lr/10
        else:
            return lr/1000 


    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler),
                tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',min_delta = 1e-2,patience = 4, verbose = 1)]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-5),
                #loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
                #loss='sparse_categorical_crossentropy',
                loss = 'binary_crossentropy',
                metrics=['accuracy'])

    history = model.fit(train_data,train_label,batch_size = 16,epochs = 10,callbacks = callback,validation_split = 0.2)
  
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    model.save(save_dir+model_name)


def test(model, x,y,target_id):
    predictions = model.predict(x)
    predictions = tf.squeeze(tf.convert_to_tensor(predictions>0.7,dtype = tf.int32))
    correct = [predictions == y]
    accuracy = np.array(correct).sum()/10000
    print("the accuracy of model {} is ".format(target_id), accuracy)

def calculate_time(x,y):
    Serial_Models = Net.Get_SerialModel()
    for i in range(10):
        inputs = x[y == i]
        inputs = inputs[0:1]
        time_start = time.time()
        output,num = Net.run_SerialModel(Serial_Models,inputs) 
        time_end=time.time()      
        print('time cost for class {},{} is {}'.format(i,num,time_end-time_start))

if __name__ == "__main__":
    #load data
    Dataset = Data.dataset("cifar10")
    x_train, y_train , x_test, y_test = Dataset.load_data()

    for i in range(10):
        model_name = "VGG19_{}_noweights.h5".format(i)
        x_retrain_train, y_retrain_train , x_retrain_test, y_retrain_test = Dataset.split_data(x_train, y_train , x_test, y_test,i)
        #测试模型准确率
        isload_model = False
        if isload_model == True:
            model = tf.keras.models.load_model(save_dir+model_name)
            test(model,x_retrain_test,y_retrain_test,i)
        #训练子模型
        else:
            #model = Net.Get_VGG19()
            model = tf.keras.models.load_model("./model/noweights/"+model_name)
            train(model,x_retrain_train,y_retrain_train,model_name = "VGG19_{}_retrain.h5".format(i))
            test(model,x_retrain_test,y_retrain_test,i)

    #计算每个class的时间
    #calculate_time(x_test,y_test)

    


    