# -*- coding: utf-8 -*-
import os,sys,argparse,re,string,logging,random,glob,cv2,math,datetime
import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Dense, LSTM, merge, Lambda, GRU, Dot, Reshape, Concatenate, Flatten, Dropout, Bidirectional, BatchNormalization, Activation, UpSampling2D, Conv2D, Reshape, GlobalAveragePooling2D, TimeDistributed, Conv2DTranspose
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import min_max_norm
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LambdaCallback
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

import cv2
imageCache = {}
def loadImg(path,resize=None,useCache=True):
    """
    0-255の形で画像を返す。すでにRGBへ変換されている
    """
    if path in imageCache: return imaceCache[path]
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize: img = cv2.resize(img, resize)
    img = img.astype(np.float32)
    img = (img-128.)/255.
    return img

class net(object):
    def __init__(self,args):
        self.nDim    = args.nDim
        self.nBatch  = args.nBatch
        self.nLength = args.nLength
        self.learnRate = args.learnRate
        self.saveFolder = args.saveFolder
        self.imgFolder = args.imgFolder
        self.timeStepHours = 3
        self.imgSize = (144,144,3)
        #self.buildModel()
        return

    def loadModel(self,fPath):
        self.model.load_weights(fPath)
        print "weights loaded"
        return

    def buildModel(self):
        K.set_learning_phase(1)

        conv_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.imgSize, pooling="avg")
        # すべてのレイヤーのパラメータを固定して高速化
        #for layers in base_model.layers:
        #    layer.trainable = False

        inX = Input(shape=(self.nLength,self.imgSize[0],self.imgSize[1],self.imgSize[2]),name="inX")
        
        h = TimeDistributed(conv_model)(inX)

        h = GRU(self.nDim)(h)

        h = Dense(4*4*128)(h)
        h = Reshape((4,4,128))(h)

        h = Conv2DTranspose(64,5,strides=4,activation="relu",padding="same")(h) # 4*4=16
        h = Conv2DTranspose(32,5,strides=3,activation="relu",padding="same")(h) # 16*3=48
        h = Conv2DTranspose( 3,5,strides=3,activation=None  ,padding="same")(h) # 48*3=144

        outY = Activation("tanh",name="outY")(h)

        model = Model(inputs=inX, outputs=outY)
        model.compile(loss="mean_absolute_error", optimizer=Adam(lr=self.learnRate))

        model.summary()
        self.model = model

        return

    def getFileNum(self,verbose=False):
        imgPath = []
        for f in glob.glob(self.imgFolder+"/*.jpg"):
            imgPath.append(f)
        self.nTotalImages = len(imgPath)
        if verbose:
            print "# img : =",len(imgPath)
        return imgPath

    def dataGen(self,loadOne=False):
        imgPath = self.getFileNum(verbose=not loadOne)

        batchX = np.zeros( (self.nBatch, self.nLength, self.imgSize[0], self.imgSize[1], self.imgSize[2]))
        batchY = np.zeros( (self.nBatch,               self.imgSize[0], self.imgSize[1], self.imgSize[2]))

        batchIdx = 0
        while True:
            # 適切な予測点を見つける
            tgt = random.choice(imgPath)
            tgt_folder = os.path.dirname(tgt)
            tgt_base   = os.path.basename(tgt)
            d0  = datetime.datetime.strptime(tgt_base,"%Y_%m_%d_%H_00.jpg")
            isGood = True
            pathList = []
            for i in range(self.nLength):
                d = d0 - datetime.timedelta(hours=self.timeStepHours*(i+1)) # 3時間ごとの画像を使用
                fileName = os.path.join(tgt_folder,d.strftime("%Y_%m_%d_%H_00.jpg"))
                pathList.append(fileName)
                if not os.path.exists(fileName):
                    isGood = False
                    break
            if not isGood: continue

            for idx,path in enumerate(pathList[::-1]):
                batchX[batchIdx,idx]  = loadImg(path,resize=self.imgSize[:2])
            batchY[batchIdx]  = loadImg(tgt,resize=self.imgSize[:2])

            batchIdx+=1
            if batchIdx==self.nBatch:
                batchIdx = 0
                yield  {"inX":batchX},{"outY":batchY}
                if loadOne: raise StopIteration

    def train(self):
        if not os.path.exists(self.saveFolder):
            os.makedirs(self.saveFolder)
            os.makedirs(self.saveFolder+"/"+"images")
        self.getFileNum()

        cp_cb = ModelCheckpoint(filepath = self.saveFolder+"/weights.{epoch:02d}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
        tb_cb = TensorBoard(log_dir=self.saveFolder, histogram_freq=1)
        def func_store(epoch,logs):

            for x in self.dataGen(loadOne=True):
                pass
            x, t = x
            y = self.model.predict_on_batch(x)

            x = x["inX"]
            t = t["outY"]
            nx = x.shape[1]+2
            ny = x.shape[0]
            h  = x.shape[2]
            w  = x.shape[3]
            r  = np.zeros((h*ny,w*nx,3),dtype=np.float32)
            for iy in range(ny):
                for ix in range(nx-2):
                    r[ iy*h:(iy+1)*h, ix*w:(ix+1)*w ] = x[iy,ix]

                ix = nx-2
                r[ iy*h:(iy+1)*h, ix*w:(ix+1)*w ] = y[iy]

                ix = nx-1
                r[ iy*h:(iy+1)*h, ix*w:(ix+1)*w ] = t[iy]

            cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d.png"%epoch),cv2.cvtColor(r*256.+128.,cv2.COLOR_RGB2BGR))


        store = LambdaCallback(on_epoch_begin=func_store)

        self.model.fit_generator(generator=self.dataGen(),
                                epochs=100000000,
                                callbacks=[cp_cb,tb_cb,store],
                                steps_per_epoch=int(self.nTotalImages/self.nBatch/10),
                                use_multiprocessing=True, 
                                max_queue_size=10, 
                                workers=1)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode","-m",dest="mode",type=str,choices=["train"],default="train")
    parser.add_argument("--nBatch" ,"-b",dest="nBatch",type=int,default=10)
    parser.add_argument("--nLength","-l",dest="nLength",type=int,default=5)
    parser.add_argument("--nDim","-d",dest="nDim",type=int,default=100)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=1e-4)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="save")
    parser.add_argument("--imgFolder","-i",dest="imgFolder",type=str,default="../data/satellite")

    args = parser.parse_args()
    n = net(args)
    n.buildModel()
    n.train()
