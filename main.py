#9th
#AX-1

from pydub import AudioSegment
import os
import numpy as np
from ARSLR import slr
import random
from tqdm import tqdm

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


class Convs:#Block of convolutions
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
    
class Ter:#Block of Terminal
    def __init__(self,model,ans):
        with tf.name_scope("Terminal") as scope:
            self.Loss=tf.reduce_sum(-tf.log(model)*ans)
            self.Train=tf.train.AdamOptimizer().minimize(self.Loss)
            self.loss=self.Loss,model
            self.train=self.Train,self.Loss,model
    

if __name__ == '__main__':
    dwin=250000;datafolder="../DATABASE/AX-1/Train";evalfolder="../DATABASE/AX-1/Eval"
    folders=ffzl(datafolder);evals=ffzl(evalfolder)
    
    x=tf.compat.v1.placeholder(tf.float32, shape=(None, dwin))
    y=tf.compat.v1.placeholder(tf.int32, shape=(None,1))
    ans=tf.compat.v1.one_hot(y, len(folders))
    model=tf.reshape(x,[-1,dwin,1])
    model=Convs(16)(model)
    model=Convs(32)(model)
    model=Convs(64)(model)
    model=Convs(96)(model)
    model=KL.GlobalMaxPool1D()(model)
    model=KL.Dense(len(folders), activation='softmax')(model)
    model=Ter(model,ans)
    
    tf.summary.scalar('loss',model.Loss)
    merged=tf.summary.merge_all()
#    tf.summary.scalar('eval_loss',model.Loss)
#    mergedeval=tf.summary.merge()
    
    with tf.Session() as sess:
        with tf.summary.FileWriter('./logs', sess.graph) as writer:
            sess.run(tf.global_variables_initializer())
            SLR=slr();SLR.load(sess)
        
            #training
            for i in tqdm(range(3000)):
                #Due to the capacity of DRAM, read every use
                wx,ly=wavdt_is2np(folders,dwin)
                _,Loss,_=sess.run(model.train,feed_dict={x:np.reshape(wx,(1,dwin)),y:np.reshape(ly,(1,1))})
                
                #logging
                summary=sess.run(merged,feed_dict={x:np.reshape(wx,(1,dwin)),y:np.reshape(ly,(1,1))})
                writer.add_summary(summary,global_step=i)
                
    
            #evaluation
            for label in range(len(evals)):
                for l_id in range(len(evals[label])):1
#                    summary=sess.run(mergedeval,feed_dict
#                                       ={x:np.reshape(wavdt(evals[label][l_id],dwin),(1,dwin)),
#                                         y:np.reshape(np.array(label),(1,1))})
#                    writer.add_summary(summary,global_step=len(evals[label])*label+l_id)
#                    print("label,id=",label,l_id,":",Loss,pred)
                
            SLR.cnt+=3000
            SLR.save(sess)
        
        
        