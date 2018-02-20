import theano
import numpy as np
import _pickle as pickle
import math


def loaddict(name):
    '''Function required to load the dictionaries'''
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def batchcreation_test(datax,datam,dataid,batchsize,shuffle):
    '''Function to organize the data in batches to go through the neural network'''
    lengthdata = math.ceil(len(datax)/float(batchsize))*float(batchsize)
    indices = np.arange(lengthdata,dtype=int)
    zeroindices = lengthdata - len(datax)
    xshape=np.shape(datax)
    mshape=np.shape(datam)
    idshape=np.shape(dataid)
    xzeros=np.zeros((int(zeroindices),xshape[1],xshape[2]))
    idzeros=np.zeros((int(zeroindices),idshape[1]))
    mzeros=np.zeros((int(zeroindices),mshape[1],mshape[2]))
    xdata=np.concatenate((datax,xzeros),axis=0)
    iddata=np.concatenate((dataid,idzeros),axis=0)
    mdata=np.concatenate((datam,mzeros),axis=0)
    if shuffle:
        np.random.shuffle(indices)
        finalbatchx=[]
        finalbatchid=[]
        finalbatchm=[]
        batchindices=np.split(indices,len(indices)/batchsize)
        for bindex in batchindices:
            finalbatchx.append(xdata[bindex,:,:])
            finalbatchid.append(iddata[bindex,:])
            finalbatchm.append(mdata[bindex,:,:])
#            print(bindex,'batch done out of ',len(batchindices))
    else:
        finalbatchx=[]
        finalbatchid=[]
        finalbatchm=[]
        batchindices=np.split(indices,len(indices)/batchsize)
        for bindex in batchindices:
            finalbatchx.append(xdata[bindex,:,:])
            finalbatchid.append(iddata[bindex,:])
            finalbatchm.append(mdata[bindex,:,:])
 #           print(bindex,'batch done out of ',len(batchindices))
    return np.array(finalbatchx).astype(theano.config.floatX),np.array(finalbatchm).astype(theano.config.floatX),np.array(finalbatchid)

