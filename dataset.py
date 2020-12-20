import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from random import shuffle


class dataset():
    def __init__(self,name = "cifar10"):
        self.name = name

    def load_data(self):
        if self.name == "cifar10":
            (x_train, y_train) , (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        elif self.name == "cifar100":
            x_train, y_train , x_test, y_test = tf.keras.datasets.cifar100.load_data()
        #简单预处理
        x_train=tf.convert_to_tensor(x_train)/255
        x_test=tf.convert_to_tensor(x_test)/255
        y_train=tf.convert_to_tensor(y_train,dtype = tf.int32)
        y_test=tf.convert_to_tensor(y_test,dtype = tf.int32)

        y_train = tf.squeeze(y_train)
        y_test = tf.squeeze(y_test)
        return x_train, y_train , x_test, y_test

    #重新构建训练二分类网络的数据集
    def split_data(self,x_train, y_train , x_test, y_test,target_class):

        #构建训练数据集和测试集
        x_retrain_train= tf.fill([0,32,32,3],value = 0.)
        y_retrain_train = tf.fill([0],value = 0)
        for i in range(10):
            if i == target_class:
                positive_image = x_train[y_train == i]
                x_retrain_train = tf.concat ([x_retrain_train,positive_image],axis = 0 )
                positive_label = tf.ones_like(y_train[y_train == i])
                y_retrain_train = tf.concat([y_retrain_train,positive_label],axis = 0)
            else:
                negative_image = x_train[y_train == i]
                negative_image = tf.random.shuffle(negative_image,seed = 10)
                negative_image = negative_image[:500]
                negative_label = tf.zeros([500],tf.int32)
                x_retrain_train = tf.concat ([x_retrain_train,negative_image],axis = 0 )
                y_retrain_train = tf.concat([y_retrain_train,negative_label],axis = 0)
                # x_retrain_train = tf.random.shuffle(x_retrain_train,seed = 10)
                # y_retrain_train = tf.random.shuffle(y_retrain_train,seed = 10)
                data = list(zip(x_retrain_train,y_retrain_train))
                shuffle(data)
                x_retrain_train,y_retrain_train = zip(*data)
                x_retrain_train = tf.convert_to_tensor(x_retrain_train)
                y_retrain_train = tf.convert_to_tensor(y_retrain_train)


        x_retrain_test = x_test
        y_retrain_test = [1 if x == target_class else 0 for x in y_test]
        y_retrain_test = tf.convert_to_tensor(y_retrain_test)


        return x_retrain_train,y_retrain_train,x_retrain_test,y_retrain_test


def display(image,label):
    plt.figure(figsize=(8,8))
    for n, image in enumerate(image):
        plt.subplot(5,2,n+1)
        plt.title(str(label[n].numpy()))
        plt.imshow(image.numpy()) 
        #plt.imshow(label.astype())
    plt.show()


if __name__ == "__main__":
    Dataset = dataset("cifar10")
    x_train, y_train , x_test, y_test = Dataset.load_data()
    x_retrain_train, y_retrain_train , x_retrain_test, y_retrain_test = Dataset.split_data(x_train, y_train , x_test, y_test,0)
    x_sample =x_train[y_train == 1] 
    y_sample = y_train[y_train == 1]
    #display(x_sample[0:9],y_sample[0:9])
    display(x_retrain_train[40:49],y_retrain_train[40:49])
    pass
