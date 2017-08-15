# -*- coding: utf-8 -*-
import os,sys,argparse,re,string,logging,random,glob,cv2,math
import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Dense, LSTM, merge, Lambda, GRU, Dot, Reshape, Concatenate, Flatten, Dropout, Bidirectional, BatchNormalization, Activation, UpSampling2D, Conv2D, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import min_max_norm
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

def memorize(f):
    cache = {}
    def helper(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    return helper

class net(object):
    def __init__(self,args):
        self.nDim    = args.nDim
        self.nBatch  = args.nBatch
        self.learnRate = args.learnRate
        self.saveFolder = args.saveFolder
        self.imgFolder = args.imgFolder
        self.imgSize = (112,112,3)
        self.buildModel()
        return

    def buildModel(self):
        def Generator():
            Gen = Sequential()
            Gen.add(Dense(input_dim=self.nDim, units=1024))
            Gen.add(BatchNormalization())
            Gen.add(Activation("relu"))
            Gen.add(Dense(units=7*7*128))
            Gen.add(BatchNormalization())
            Gen.add(Activation("relu"))
            Gen.add(Reshape((7,7,128), input_shape=(7*7*128,)))

            Gen.add(UpSampling2D((2,2)))
            Gen.add(Conv2D(64,5, padding="same"))
            Gen.add(BatchNormalization())
            Gen.add(Activation("relu"))

            Gen.add(UpSampling2D((2,2)))
            Gen.add(Conv2D(32,5, padding="same"))
            Gen.add(BatchNormalization())
            Gen.add(Activation("relu"))

            Gen.add(UpSampling2D((2,2)))
            Gen.add(Conv2D(8, 5, padding="same"))
            Gen.add(BatchNormalization())
            Gen.add(Activation("relu"))

            Gen.add(UpSampling2D((2,2)))
            Gen.add(Conv2D(3, 5, padding="same"))
            Gen.add(Activation("tanh"))
            #generator_optimizer = SGD(lr=0.1, momentum=0.3, decay=1e-5)
            #Gen.compile(loss="binary_crossentropy", optimizer=generator_optimizer)
            return Gen

        def Discriminator():
            act = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)
            Dis = Sequential()
            Dis.add(Conv2D(filters= 8, kernel_size=(5, 5), strides=(2,2), padding="same", input_shape=(112,112,3)))
            Dis.add(act)
            Dis.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2,2), padding="same"))
            Dis.add(act)
            Dis.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(2,2), padding="same"))
            Dis.add(act)
            Dis.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(2,2), padding="same"))
            Dis.add(act)
            Dis.add(Flatten())
            Dis.add(Dense(units=1024))
            Dis.add(act)
            Dis.add(Dropout(0.5))
            Dis.add(Dense(1))
            Dis.add(Activation("sigmoid"))
            discriminator_optimizer = Adam(lr=1e-5, beta_1=0.1)
            Dis.compile(loss="binary_crossentropy", optimizer=discriminator_optimizer)
            return Dis

        self.model_gen = model_gen = Generator()
        self.model_dsc = model_dsc = Discriminator()

        model = Sequential()

        model.add(model_gen)

        #model_dsc.trainable=False
        model.add(model_dsc)
        model_optimizer = Adam(lr=1e-5, beta_1=0.1)

        model.compile(loss="binary_crossentropy", optimizer=model_optimizer)

        model.summary()
        self.model = model
        print "done"

        return

    @memorize
    def loadImg(self,path):
        img = cv2.imread(path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgSize[0], self.imgSize[1]))
        img = img.astype(np.float32)
        img = (img-128.)/128. # -1~+1の範囲にしようと思ったら、本当は128で割るのが正しい。しかしここではあえて極端な値から離れる意味で、-0.5~+-0.5の範囲に規格化する
        #print img.max(), img.min()
        return img


    def dataGen(self):
        imgPath = []
        for f in glob.glob(self.imgFolder+"/*.jpg"):
            imgPath.append(f)
        print "# img : =",len(imgPath)
        while True:
            imgBatch = np.zeros( (self.nBatch, self.imgSize[0], self.imgSize[1], self.imgSize[2]))
            for idx,path in enumerate(random.sample(imgPath,self.nBatch)):
                imgBatch[idx]  = self.loadImg(path)
            yield imgBatch


    def train(self):
        if not os.path.exists(self.saveFolder):
            os.makedirs(self.saveFolder)
            os.makedirs(self.saveFolder+"/"+"images")

        step = 0
        for data in self.dataGen():
            z     = np.random.uniform(0,1,size=(self.nBatch, self.nDim))
            x_gen = self.model_gen.predict_on_batch(z)
            x_ori = data

            x_com = np.vstack( (x_ori, x_gen) )
            t_com = np.zeros( 2 * self.nBatch , dtype=np.int32)
            t_com[self.nBatch:] = 1 # 生成されたものが1、自然な画像が0になるように設定

            self.model_dsc.trainable = True
            d_loss = self.model_dsc.train_on_batch(x=x_com, y=t_com)
            self.model_dsc.trainable = False

            z     = np.random.uniform(0,1,size=(self.nBatch, self.nDim))
            t_gen = np.zeros( self.nBatch, dtype=np.int32 ) # 自然な画像が0になるようにするので、これはすべて0としておいて、自然な画像が生成される方向に最適化を向かわせる
            g_loss = self.model.train_on_batch(x=z, y=t_gen)

            """
            # 2回繰り返すと安定するという噂
            z     = np.random.uniform(0,1,size=(self.nBatch, self.nDim))
            t_gen = np.zeros( self.nBatch, dtype=np.int32 ) # 自然な画像が0になるようにするので、これはすべて0としておいて、自然な画像が生成される方向に最適化を向かわせる
            g_loss = self.model.train_on_batch(x=z, y=t_gen)
            """

            def tileImage(imgs):
                d = int(math.sqrt(imgs.shape[0]-1))+1
                h = imgs[0].shape[0]
                w = imgs[0].shape[1]
                r = np.zeros((h*d,w*d,3),dtype=np.float32)
                for idx,img in enumerate(imgs):
                    idx_y = int(idx/d)
                    idx_x = idx-idx_y*d
                    r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
                return r

            step += 1
            if step%100==0:
                print "%5d: gen_loss=%.3e dsc_loss=%.3e"%(step,g_loss, d_loss)
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_ori.png"%step),tileImage(x_ori)*128.+128.)
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_gen.png"%step),tileImage(x_gen)*128.+128.)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode","-m",dest="mode",type=str,choices=["train"],default="train")
    parser.add_argument("--nBatch" ,"-b",dest="nBatch",type=int,default=100)
    parser.add_argument("--nLength","-l",dest="nLength",type=int,default=10)
    parser.add_argument("--nDim","-d",dest="nDim",type=int,default=100)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=1e-4)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="save")
    parser.add_argument("--imgFolder","-i",dest="imgFolder",type=str,default="../data/satellite")

    args = parser.parse_args()
    n = net(args)
    n.train()
