#9th
#AX-1

from pydub import AudioSegment
import os
import numpy as np
from ARSLR import slr#a module I made
import random
from tqdm import tqdm
import argparse

import tensorflow as tf
import tensorflow.python.keras.layers as KL
import tensorflow.python.keras as K
from tensorboard.plugins.hparams import api as hp

#wavedata to numpy
def wavdt(input,dwin):
    data=AudioSegment.from_file(input,"wav")
    if data.frame_rate!=44100:raise print(data.frame_rate,"[Hz]:The frame rate is incorrect")
    data2=np.array(data.get_array_of_samples()[::data.channels],"float32")
    #Padding if the wavedata does not have enough size.
    if dwin-len(data2)>0:data2=np.pad(data2, (0, dwin-len(data2)), 'constant', constant_values=0.0)
    return data2[:dwin]*2.0/(256**data.frame_width)

#input[label][id in label]>numpyfile,label(wavdt wrapper)
def wavdt_is2np(input,dwin):
    t_label=random.randrange(len(input))
    t_id=random.randrange(len(input[t_label]))
    return wavdt(input[t_label][t_id],dwin),np.array(t_label)

#grasp dataset directory structure. image_struct[label][id in label]=data directory
def ffzl(input):
    image_struct=[]
    for fd_path, sb_folder, sb_file in os.walk(input):
        if 0<len(sb_file):
            image_struct.append([])
            for fil in sb_file:image_struct[-1].append(fd_path + '/' + fil)
    return image_struct

#Block of convolutions
class Convs:
    def __init__(self,dim=512):
        self.dim=dim
    def __call__(self,model):
        with tf.name_scope("Convs") as scope:
            sc=KL.MaxPooling1D(8)(KL.Conv1D(self.dim*3,8,activation='relu',padding="same")(model))
            model=KL.MaxPooling1D(2)(KL.Conv1D(self.dim,4,activation='relu',padding="same")(model))
            model=KL.MaxPooling1D(2)(KL.Conv1D(self.dim*2,4,activation='relu',padding="same")(model))
            model=KL.MaxPooling1D(2)(KL.Conv1D(self.dim*3,4,activation='relu',padding="same")(model))
            model=KL.add([model, sc])
            model=KL.Dropout(0.25)(model)
        return model
    
#Block of Terminal
class Ter:
    def __init__(self,model,ans):
        with tf.name_scope("Terminal") as scope:
            self.Loss=tf.reduce_sum(-tf.log(model)*ans)
            self.Train=tf.train.AdamOptimizer().minimize(self.Loss)
            self.loss=self.Loss,model
            self.train=self.Train,self.Loss,model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='AX-1',
            description='This is a classification sample of actors voice by TF.',
            add_help=True
            )
    parser.add_argument('--train', help='train data dir',default="../DATABASE/AX-1/Train")
    parser.add_argument('--test', help='test data dir',default="../DATABASE/AX-1/Test")
    parser.add_argument('--dwin', help='size of wavefile(dwin/44100[s])',default=250000,type=int)
    args = parser.parse_args(args=[])
    dwin=args.dwin;
    folders=ffzl(args.train);tests=ffzl(args.test)
    
    x=tf.compat.v1.placeholder(tf.float32, shape=(None, dwin))
    y=tf.compat.v1.placeholder(tf.int32, shape=(None,1))
    ans=tf.compat.v1.one_hot(y, len(folders))
    model=tf.reshape(x,[-1,dwin,1])
    model=Convs(16)(model)
    model=Convs(32)(model)
    model=Convs(64)(model)
    model=KL.GlobalMaxPool1D()(model)
    model=KL.Dropout(0.25)(model)
    model=KL.Dense(len(folders), activation='softmax')(model)
    model=Ter(model,ans)
    
    mg_loss=tf.compat.v1.summary.scalar('loss',model.Loss)
    mg_test=tf.compat.v1.summary.scalar('test',model.Loss)
    
    with tf.Session() as sess:
        with tf.summary.FileWriter('./logs', sess.graph) as writer:
            sess.run(tf.global_variables_initializer())
            SLR=slr();SLR.load(sess)
        
            #training
            for i in tqdm(range(6000)):
                #Due to the capacity of DRAM, read every use
                wx,ly=wavdt_is2np(folders,dwin)
                result=sess.run([model.train,mg_loss],feed_dict={x:np.reshape(wx,(1,dwin)),
                                                                 y:np.reshape(ly,(1,1))})
                summary=result[1]
                #logging
                writer.add_summary(summary,global_step=i)
                
            #evaluation
            
            for label in range(len(tests)):
                for l_id in range(len(tests[label])):
                    summary=sess.run(mg_test,feed_dict
                                       ={x:np.reshape(wavdt(tests[label][l_id],dwin),(1,dwin)),
                                         y:np.reshape(np.array(label),(1,1))})
                    writer.add_summary(summary,global_step=len(tests[label])*label+l_id)
            
            SLR.cnt+=6000
            SLR.save(sess)
        
        
        