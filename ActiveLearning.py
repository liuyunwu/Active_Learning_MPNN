#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle as pkl
#from sklearn.preprocessing import StandardScaler
import csv, sys


# In[2]:


import numpy as np
import tensorflow as tf
import sys, time, warnings
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors  
from sklearn.metrics import mean_absolute_error
from rdkit.Chem import AllChem


# In[3]:


from rdkit.DataStructs import FingerprintSimilarity
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.Fingerprints import FingerprintMols


# In[4]:


import sys
import os
import math

import numpy
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdmd

from sklearn.cluster import MiniBatchKMeans

import pandas as pd
from tqdm import tqdm
import time
import numpy as np

from scipy.spatial.distance import cdist

#from docopt import docopt


# In[5]:


class Model(object):

    def __init__(self, n_node, dim_node, dim_edge, dim_atom, dim_y, dim_h=64, n_mpnn_step=5, dr=0.2, batch_size=20, lr=0.0001, useGPU=True):

        warnings.filterwarnings('ignore')
        tf.logging.set_verbosity(tf.logging.ERROR)
        rdBase.DisableLog('rdApp.error') 
        rdBase.DisableLog('rdApp.warning')

        self.n_node=n_node
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.dim_atom=dim_atom
        self.dim_y=dim_y

        self.dim_h=dim_h
        self.n_mpnn_step=n_mpnn_step
        self.dr=dr
        self.batch_size=batch_size
        self.lr=lr

        # variables
        self.G = tf.Graph()
        self.G.as_default()

        self.trn_flag = tf.placeholder(tf.bool)
        
        self.node = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.dim_node])
        self.edge = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.n_node, self.dim_edge])      
        self.proximity = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.n_node, 1])
        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])
        
        self.hidden_0, self.hidden_n = self._MP(self.batch_size, self.node, tf.concat([self.edge, self.proximity], 3), self.n_mpnn_step, self.dim_h)
        self.Y_pred = self._Readout(self.batch_size, self.node, self.hidden_0, self.hidden_n, self.dim_h * 4, self.dim_y, self.dr)
                 
        # session
        self.saver = tf.train.Saver()
        if useGPU:
            self.sess = tf.Session()
        else:
            config = tf.ConfigProto(device_count = {'GPU': 0} )
            self.sess = tf.Session(config=config)
             
        
    def train(self, DV_trn, DE_trn, DP_trn, DY_trn, DV_val, DE_val, DP_val, DY_val, update=False):

        ## objective function
        cost_Y = tf.reduce_mean(tf.square(self.Y - self.Y_pred))

        vars_MP = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MP')
        vars_Y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Y')

        assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) == len(vars_Y) + len(vars_MP)
        
        train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost_Y)
        
        self.sess.run(tf.initializers.global_variables())            
        np.set_printoptions(precision=5, suppress=True)

        n_batch = int(len(DV_trn)/self.batch_size)

        #if load_path is not None:
         #   self.saver.restore(self.sess, load_path)
            
        ## tranining
        if update:
            max_epoch = 10
            
        else:
            max_epoch = 50
        #self.saver.save(self.sess, save_path)
        
        print('::: training')
        trn_log = np.zeros(max_epoch)
        val_t = np.zeros(max_epoch)

        for epoch in range(max_epoch):

            # training
            [DV_trn, DE_trn, DP_trn, DY_trn] = self._permutation([DV_trn, DE_trn, DP_trn, DY_trn])

            trnscores = np.zeros(n_batch) 
            if epoch > 0:
                for i in range(n_batch):

                    start_=i*self.batch_size
                    end_=start_+self.batch_size

                    assert self.batch_size == end_ - start_

                    trnresult = self.sess.run([train_op, cost_Y],
                                                  feed_dict = {self.node: DV_trn[start_:end_], self.edge: DE_trn[start_:end_], 
                                                               self.proximity: DP_trn[start_:end_], self.Y: DY_trn[start_:end_], self.trn_flag: True}) 

                    trnscores[i] = trnresult[1]

                trn_log[epoch] = np.mean(trnscores)        
                #print('--training yid: ', yid, ' epoch id: ', epoch, ' trn log: ', trn_log[epoch])

            # validation
            DY_val_hat = self.test(DV_val, DE_val, DP_val)
            val_mae = mean_absolute_error(DY_val, DY_val_hat)
            val_t[epoch] = val_mae
            #print('--evaluation yid: ', yid, ' epoch id: ', epoch, ' val MAE: ', val_t[epoch], 'BEST: ', np.min(val_t[0:epoch+1]), np.min(val_t[0:epoch+1])/self.dim_y)
            #print('--evaluation yid: ', yid, ' list: ', val_mae)         

            if epoch > 20 and np.min(val_t[0:epoch-20]) < np.min(val_t[epoch-20:epoch+1]):
                print('--termination condition is met')
                break

            #elif np.min(val_t[0:epoch+1]) == val_t[epoch]:
            #    self.saver.save(self.sess, save_path) 


    def test(self, DV_tst, DE_tst, DP_tst, trn_flag=False):
    
        n_batch_tst = int(len(DV_tst)/self.batch_size)
        DY_tst_hat=[]
        for i in range(n_batch_tst):
        
            start_=i*self.batch_size
            end_=start_+self.batch_size
            
            assert self.batch_size == end_ - start_
            
            DY_tst_batch = self.sess.run(self.Y_pred,
                                         feed_dict = {self.node: DV_tst[start_:end_], self.edge: DE_tst[start_:end_],
                                                      self.proximity: DP_tst[start_:end_], self.trn_flag: trn_flag})
            DY_tst_hat.append(DY_tst_batch)
        
        DY_tst_hat = np.concatenate(DY_tst_hat, 0)

        return DY_tst_hat      


    def _permutation(self, set):
    
        permid = np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i] = set[i][permid]
    
        return set
        
    
    def _MP(self, batch_size, node, edge, n_step, hiddendim):

        def _embed_node(inp):
        
            inp = tf.layers.dense(inp, hiddendim * 4, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 4, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 4, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim, activation = tf.nn.tanh)
        
            inp = inp * mask
        
            return inp

        def _edge_nn(inp):
        
            inp = tf.layers.dense(inp, hiddendim * 4, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 4, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 4, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * hiddendim)
        
            inp = tf.reshape(inp, [batch_size, self.n_node, self.n_node, hiddendim, hiddendim])
            inp = inp * tf.reshape(1-tf.eye(self.n_node), [1, self.n_node, self.n_node, 1, 1])
            inp = inp * tf.reshape(mask, [batch_size, self.n_node, 1, 1, 1]) * tf.reshape(mask, [batch_size, 1, self.n_node, 1, 1])

            return inp

        def _MPNN(edge_wgt, node_hidden, n_step):
        
            def _msg_nn(wgt, node):
            
                wgt = tf.reshape(wgt, [batch_size * self.n_node, self.n_node * hiddendim, hiddendim])
                node = tf.reshape(node, [batch_size * self.n_node, hiddendim, 1])
            
                msg = tf.matmul(wgt, node)
                msg = tf.reshape(msg, [batch_size, self.n_node, self.n_node, hiddendim])
                msg = tf.transpose(msg, perm = [0, 2, 3, 1])
                msg = tf.reduce_mean(msg, 3)
            
                return msg

            def _update_GRU(msg, node, reuse_GRU):
            
                with tf.variable_scope('mpnn_gru', reuse=reuse_GRU):
            
                    msg = tf.reshape(msg, [batch_size * self.n_node, 1, hiddendim])
                    node = tf.reshape(node, [batch_size * self.n_node, hiddendim])
            
                    cell = tf.nn.rnn_cell.GRUCell(hiddendim)
                    _, node_next = tf.nn.dynamic_rnn(cell, msg, initial_state = node)
            
                    node_next = tf.reshape(node_next, [batch_size, self.n_node, hiddendim]) * mask
            
                return node_next

            nhs=[]
            for i in range(n_step):
                message_vec = _msg_nn(edge_wgt, node_hidden)
                node_hidden = _update_GRU(message_vec, node_hidden, reuse_GRU=(i!=0))
                nhs.append(node_hidden)
        
            out = tf.concat(nhs, axis=2)
            
            return out

        with tf.variable_scope('MP', reuse=False):
        
            mask = tf.reduce_max(node[:,:,:self.dim_atom], 2, keepdims=True)
            
            edge_wgt = _edge_nn(edge)
            hidden_0 = _embed_node(node)
            hidden_n = _MPNN(edge_wgt, hidden_0, n_step)
            
        return hidden_0, hidden_n


    def _Readout(self, batch_size, node, hidden_0, hidden_n, aggrdim, ydim, drate):
      
        def _readout(hidden_0, hidden_n, outdim):    
            
            def _attn_nn(inp, hdim):
            
                inp = tf.layers.dense(inp, hdim, activation = tf.nn.sigmoid)
                
                return inp
        
            def _tanh_nn(inp, hdim):
            
                inp = tf.layers.dense(inp, hdim)
            
                return inp

            attn_wgt = _attn_nn(tf.concat([hidden_0, hidden_n], 2), aggrdim) 
            tanh_wgt = _tanh_nn(hidden_n, aggrdim)
            readout = tf.reduce_mean(tf.multiply(tanh_wgt, attn_wgt) * mask, 1)
            for _ in range(3):
                readout = tf.layers.dense(readout, aggrdim, activation = tf.nn.relu)
                readout = tf.layers.dropout(readout, drate, training = self.trn_flag)
                
            pred = tf.layers.dense(readout, outdim) 
    
            return pred

        mask = tf.reduce_max(node[:,:,:self.dim_atom], 2, keepdims=True)

        with tf.variable_scope('Y', reuse=False):

            readout = _readout(hidden_0, hidden_n, 1)

        return rout


# In[6]:


# main hyperparameters
n_max=29
dim_node=13 
dim_edge=4

atom_list=['H','C','N','O','F'] 
hybridization_list = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]

data_path = './QM9_graph.pkl'
save_path = './MPNN_model.ckpt'

print(':: load data')
with open('QM9_Graph.pkl','rb') as f:
    [DV, DE, DP, DY, Dsmi] = pkl.load(f)

DV = DV.todense()
DE = DE.todense()
DP = np.expand_dims(DP, 3)
DY = DY[:, 0:1]

#Standarize
scaler = StandardScaler()
DY = scaler.fit_transform(DY)

dim_atom = len(atom_list)
dim_y = DY.shape[1]

print(DV.shape, DE.shape, DP.shape, DY.shape)

print(':: preprocess data')

np.random.seed(134)
[DV, DE, DP, DY, Dsmi] = _permutation([DV, DE, DP, DY, Dsmi])

n_tst = 10000
n_val = 10000
n_init = 10000
n_U = len(DV) - n_tst - n_val - n_init


DV_L = DV[:n_trn]
DE_L = DE[:n_trn]
DP_L = DP[:n_trn]
DY_L = DY[:n_trn]
    
DV_val = DV[n_trn:n_trn+n_val]
DE_val = DE[n_trn:n_trn+n_val]
DP_val = DP[n_trn:n_trn+n_val]
DY_val = DY[n_trn:n_trn+n_val]



#Number of data
print('train data size', DV_trn.shape, DE_trn.shape, DP_trn.shape, DY_trn.shape)
print('validation data size', DV_val.shape, DE_val.shape, DP_val.shape, DY_val.shape)


# In[ ]:


# main hyperparameters
n_query = 200
n_est = 10
sim_thr = 0.3 #if 1, equivalent to simple top k selection

n_max=29
dim_node=13 
dim_edge=4

atom_list=['H','C','N','O','F'] 
hybridization_list = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]

#save_dict = './'
load_path = './MPNN_init_model.ckpt'
#save_path=save_dict+'MPNN_model.ckpt'

print(':: load data')
with open('QM9_val.pkl','rb') as f:
    [DV_val, DE_val, DP_val, DY_val, Dsmi_val] = pkl.load(f)
    
print(':: load data')
with open('QM9_Label.pkl','rb') as f:
    [DV_L, DE_L, DP_L, DY_L, Dsmi_L] = pkl.load(f)
    
with open('QM9_Unlabel.pkl','rb') as f:
    [DV_U, DE_U, DP_U, DY_U, Dsmi_U] = pkl.load(f)
    
with open('QM9_Unlabel.pkl','rb') as f:
    [DV_U, DE_U, DP_U, DY_U, Dsmi_U] = pkl.load(f)

with open('QM9_tst.pkl','rb') as f:
    [DV_tst, DE_tst, DP_tst, DY_tst, Dsmi_tst] = pkl.load(f)

DV_val = DV_val.todense()
DE_val = DE_val.todense()
DP_val = np.expand_dims(DP_val, 3)

DV_L = DV_L.todense()
DE_L = DE_L.todense()
DP_L = np.expand_dims(DP_L, 3)

DV_U = DV_U.todense()
DE_U = DE_U.todense()
DP_U = np.expand_dims(DP_U, 3)

DV_tst = DV_tst.todense()
DE_tst = DE_tst.todense()
DP_tst = np.expand_dims(DP_tst, 3)

dim_atom = len(atom_list)
dim_y = 12

#Number of data
print('Label data size', DV_L.shape, DE_L.shape, DP_L.shape, DY_L.shape)
print('Unlabel data size', DV_U.shape, DE_U.shape, DP_U.shape, DY_U.shape)
print('validation data size', DV_val.shape, DE_val.shape, DP_val.shape, DY_val.shape)
print('test data size', DV_tst.shape, DE_tst.shape, DP_tst.shape, DY_tst.shape)

print(':: preprocess data')

def _permutation(set):
    permid = np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i] = set[i][permid]

    return set

np.random.seed(100)
[DV_L, DE_L, DP_L, DY_L, Dsmi_L] = _permutation([DV_L, DE_L, DP_L, DY_L, Dsmi_L])

np.random.seed(100)
[DV_U, DE_U, DP_U, DY_U, Dsmi_U] = _permutation([DV_U, DE_U, DP_U, DY_U, Dsmi_U])

np.random.seed(100)
[DV_val, DE_val, DP_val, DY_val, Dsmi_val] = _permutation([DV_val, DE_val, DP_val, DY_val, Dsmi_val])

np.random.seed(100)
[DV_tst, DE_tst, DP_tst, DY_tst, Dsmi_tst] = _permutation([DV_tst, DE_tst, DP_tst, DY_tst, Dsmi_tst])


# In[ ]:


# main hyperparameters
n_query = 200
n_est = 10
sim_thr = 0.3 #if 1, equivalent to simple top k selection

n_max=29
dim_node=13 
dim_edge=4

atom_list=['H','C','N','O','F'] 
hybridization_list = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]

#save_dict = './'
load_path = './MPNN_init_model.ckpt'
#save_path=save_dict+'MPNN_model.ckpt'

print(':: load data')
with open('QM9_val.pkl','rb') as f:
    [DV_val, DE_val, DP_val, DY_val, Dsmi_val] = pkl.load(f)
    
print(':: load data')
with open('QM9_Label.pkl','rb') as f:
    [DV_L, DE_L, DP_L, DY_L, Dsmi_L] = pkl.load(f)
    
with open('QM9_Unlabel.pkl','rb') as f:
    [DV_U, DE_U, DP_U, DY_U, Dsmi_U] = pkl.load(f)
    
with open('QM9_Unlabel.pkl','rb') as f:
    [DV_U, DE_U, DP_U, DY_U, Dsmi_U] = pkl.load(f)

with open('QM9_tst.pkl','rb') as f:
    [DV_tst, DE_tst, DP_tst, DY_tst, Dsmi_tst] = pkl.load(f)

DV_val = DV_val.todense()
DE_val = DE_val.todense()
DP_val = np.expand_dims(DP_val, 3)

DV_L = DV_L.todense()
DE_L = DE_L.todense()
DP_L = np.expand_dims(DP_L, 3)

DV_U = DV_U.todense()
DE_U = DE_U.todense()
DP_U = np.expand_dims(DP_U, 3)

DV_tst = DV_tst.todense()
DE_tst = DE_tst.todense()
DP_tst = np.expand_dims(DP_tst, 3)

dim_atom = len(atom_list)
dim_y = 12

#Number of data
print('Label data size', DV_L.shape, DE_L.shape, DP_L.shape, DY_L.shape)
print('Unlabel data size', DV_U.shape, DE_U.shape, DP_U.shape, DY_U.shape)
print('validation data size', DV_val.shape, DE_val.shape, DP_val.shape, DY_val.shape)
print('test data size', DV_tst.shape, DE_tst.shape, DP_tst.shape, DY_tst.shape)

print(':: preprocess data')

def _permutation(set):
    permid = np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i] = set[i][permid]

    return set

np.random.seed(100)
[DV_L, DE_L, DP_L, DY_L, Dsmi_L] = _permutation([DV_L, DE_L, DP_L, DY_L, Dsmi_L])

np.random.seed(100)
[DV_U, DE_U, DP_U, DY_U, Dsmi_U] = _permutation([DV_U, DE_U, DP_U, DY_U, Dsmi_U])

np.random.seed(100)
[DV_val, DE_val, DP_val, DY_val, Dsmi_val] = _permutation([DV_val, DE_val, DP_val, DY_val, Dsmi_val])

np.random.seed(100)
[DV_tst, DE_tst, DP_tst, DY_tst, Dsmi_tst] = _permutation([DV_tst, DE_tst, DP_tst, DY_tst, Dsmi_tst])


# In[ ]:


method='random'


# In[ ]:


#Active Learning
MAE=[]
MAE_av=[]

print(':: start active learning')
model = Model(n_max, dim_node, dim_edge, dim_atom, dim_y, dim_h=32, n_mpnn_step=3, dr=0.5, lr=0.001)
with model.sess:
    
    for iteration in range(100):
    
        print(':: iter ', iteration)

        if iteration == 0:
            # model init
            #[DV_trn, DE_trn, DP_trn, DY_trn], [DV_val, DE_val, DP_val, DY_val] = _split([DV_L, DE_L, DP_L, DY_L], 0.2)
            try:
                model.saver.restore(model.sess, load_path)  
            except:
                model.train(DV_L, DE_L, DP_L, DY_L, DV_val, DE_val, DP_val, DY_val, update=False)
                model.saver.save(model.sess, load_path)
        
        else:       
            # uncertainty estimation
            if method in ['simpleK','diversifiedK']:
                DY_U_hat=[]
                for _ in range(n_est):
                    DY_U_hat.append(model.test(DV_U, DE_U, DP_U, trn_flag=True))
  
                DY_U_conf = np.mean(np.std(DY_U_hat,0), 1) # 데이터셋 불확실성 측정
                DY_U_hat=np.mean(DY_U_hat, 0)
        
                #Uncertainty Score-Absolute error scatter plot
                DY_U=DY_U[:129080]
                aelist_U = np.array(abs(DY_U-DY_U_hat))
                ae_U = np.sum(aelist_U, axis=1)
                ae_U_av=ae_U/12
                     
                #plt.scatter(DY_U_conf, ae_U_av, color='r', alpha=0.5)
                #plt.xlim(0,0.25)
                #plt.ylim(0,1.6)
                #plt.xlabel('Uncertainty')
                #plt.ylabel('Absolute Error')
                #plt.show()
        
        
            # query sampling
            if method=='random':
                ## random sampling
                query_id = np.random.permutation(len(DV_U))[:n_query]
                
            elif method=='simpleK':            
                ## simple top k selection
                query_id = np.argsort(-DY_U_conf)[:n_query]
                
            else:
                raise
                

            # data update
            assert len(query_id) == n_query
            
            DV_query = DV_U[query_id]
            DE_query = DE_U[query_id]
            DP_query = DP_U[query_id]
            DY_query = DY_U[query_id]
            
            DV_U = np.delete(DV_U, query_id, 0)
            DE_U = np.delete(DE_U, query_id, 0)
            DP_U = np.delete(DP_U, query_id, 0)
            DY_U = np.delete(DY_U, query_id, 0)
            Dsmi_U = np.delete(Dsmi_U, query_id, 0)
        
            DV_L = np.concatenate([DV_L, DV_query], 0)
            DE_L = np.concatenate([DE_L, DE_query], 0)
            DP_L = np.concatenate([DP_L, DP_query], 0)
            DY_L = np.concatenate([DY_L, DY_query], 0)
            

            
            # model update
            #[DV_trn, DE_trn, DP_trn, DY_trn], [DV_val, DE_val, DP_val, DY_val] = _split([DV_L, DE_L, DP_L, DY_L], 0.2) 
            model.train(DV_tst, DE_tst, DP_tst, DY_tst, DV_val, DE_val, DP_val, DY_val, update=True)


        # evaluation
        DY_tst_hat = model.test(DV_tst, DE_tst, DP_tst)
        maelist = np.array([mean_absolute_error(DY_tst[:,yid:yid+1], DY_tst_hat[:,yid:yid+1]) for yid in range(dim_y)])
        mae = np.sum(maelist)
        mae_av=mae/12
        
        MAE.append(mae)
        MAE_av.append(mae_av)
        
        print(':::: no. currently labeled ', len(DV_L)) 
        print(':::: test MAElist', maelist) 
        print(':::: test MAE ', mae/12) 
    
with open( 'QM9_simple', 'w') as output:
    out = csv.writer(output)
    out.writerows(map(lambda x: [x], MAE_av))

