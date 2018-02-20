import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import _pickle as pickle
from Bio import SeqIO
import argparse
##
import utils
import neural_netGRU as config
##
parser = argparse.ArgumentParser()
parser.add_argument('-in', '--input_fasta',  help="fasta file with protein sequences")
parser.add_argument('-p', '--parameters_weights', help="dictionary containing the preprocessing values and neural network weights (without .pkl)")
parser.add_argument('-o','--output_file', help="name of the output file with the IDs, aminoacid sequences, and energies")
parser.add_argument('-s','--strange_prot_file', help="name of the output file with the strange proteins not transformed and predicted")
args = parser.parse_args()

parameters=utils.loaddict(args.parameters_weights)
hyper_parameters = [0.0019146969188368781, 3, 253, 32, 0.2, 0.1, 60, 40,120,100, 100]

aminoacid_dict=utils.loaddict('./aminoacid_dictionary')


def create_dataset(fasta_file,dict_aa):
    '''Creates the dataset that will be introduced in the neural network'''
    start_time=time.time()
    print('Loading fasta file')
    record_dict = SeqIO.index(fasta_file, "fasta")
    print('Loaded fasta file took',time.time()-start_time)
    list_failed_prot=[]
    list_aa_ar=[]
    list_aa_let=[]
    list_id=[]
    length_prot=[]
    print('Enter first loop data creation')
    for i in record_dict.keys():
        aa_let=list(record_dict[i].seq)
        aa_ar=[]
        strange=False
        for a in aa_let:
            if a not in dict_aa or len(aa_let)>15000:
                strange=True
            else:
                aa_ar.append(dict_aa[a])
        aa_ar=np.array(aa_ar)
        if strange:
            list_failed_prot.append(i)
            continue
        else:
            list_aa_ar.append(aa_ar)
            list_aa_let.append(np.array(aa_let))
            list_id.append(i)
            length_prot.append(len(aa_let))
#        print(i,'out of ',record_dict.keys(),' of first loop')
    print('Exit first loop')
    max_len=np.max(length_prot)
    final_id=[]
    final_aminoacid=[]
    final_aa_bi=[]
    final_mask=[]
    print('Enter second loop')
    for i in range(len(list_id)):
        diff_len=max_len-length_prot[i]
        if diff_len!=0:
            complete_aa_ar=np.concatenate((list_aa_ar[i],np.zeros((diff_len,20))))
            mask=np.concatenate((np.ones(length_prot[i]),np.zeros(diff_len)))
            aa_amino=np.concatenate((list_aa_let[i],['0']*diff_len))
        else:
            complete_aa_ar=list_aa_ar[i]
            mask=np.ones(length_prot[i])
            aa_amino=list_aa_let[i]
        final_id.append(list_id[i])
        final_aminoacid.append(list_aa_let[i])
        final_aa_bi.append(complete_aa_ar)
        final_mask.append(np.expand_dims(mask,axis=1))
        print(i,'out of ',' of second loop')
    print('Exit second loop')
    return np.expand_dims(final_aminoacid,axis=1),np.expand_dims(np.array(final_id),axis=1),np.array(final_aa_bi),np.array(final_mask),list_failed_prot

aminoacid,id,x,mask,strange_prot=create_dataset(args.input_fasta,aminoacid_dict)

strange_file = open('./predicted_proteins/strange_proteins/'+str(args.strange_prot_file)+'.txt', 'w')
for item in strange_prot:
    strange_file.write("%s\n" % item)
strange_file.close()

def prediction_neuralnet(testx,testm,testid,parameters,hyper_parameters,neural_net):
    '''Function calling the structure of the neural network'''
    def preprocess(test,mean,std):
        test_n=test-mean
        test_n=test_n/std
        return test_n

    def predict(hyperparameters,parameters,testx,testm,testid,neural_net):
        learning_rate=hyperparameters[0]
        grad_clip=int(hyperparameters[1])
        n_hid_rec=int(hyperparameters[2])
        size_batch=int(hyperparameters[3])
        drop_perm=hyperparameters[4]
        drop_hid=hyperparameters[5]
        pre_hidden_1=int(hyperparameters[6])
        pre_hidden_2=int(hyperparameters[7])
        post_hidden_1=int(hyperparameters[8])
        post_hidden_2=int(hyperparameters[9])
        post_hidden_3=int(hyperparameters[10])
        max_len=np.shape(testx)[1]
        target_values = T.tensor3('target_output')
        mask_var = T.matrix('mask')
        input_var = T.tensor3('inputs')
        all_params = parameters
        network = config.neuralnet(pre_hidden_1=pre_hidden_1,pre_hidden_2=pre_hidden_2,drop_hid=drop_hid,post_hidden_1=post_hidden_1,post_hidden_2=post_hidden_2,post_hidden_3=post_hidden_3,n_hid_rec=n_hid_rec,grad_clip=grad_clip,drop_perm=drop_perm,size_batch=size_batch,max_length=max_len,n_features=20,input_var=input_var,mark_var=mask_var)
        lasagne.layers.set_all_param_values(network, all_params)
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        predicted_values = np.squeeze(test_prediction)
        predict_value = theano.function(inputs=[input_var,mask_var],outputs=[predicted_values,mask_var])
        btestx,btestm,btestid = utils.batchcreation_test(testx,testm,testid,size_batch,shuffle=False)
        results_mask = []
        results_pred = []
        results_id = []
        for i in range(len(btestx)):
            start_time=time.time()
            prediction,mask= predict_value(btestx[i],np.squeeze(btestm[i]))
            zeros_minus=np.sum(np.squeeze(btestid[i])=='0.0')
            if zeros_minus== 0:
                results_pred.extend(prediction)
                results_id.extend(btestid[i])
                results_mask.extend(mask)
            else:
                results_pred.extend(prediction[:-zeros_minus,:,:])
                results_id.extend(btestid[i][:-zeros_minus,:])
                results_mask.extend(mask[:-zeros_minus,:])
            print(len(results_pred),' proteins out of ',len(testx),' of the ',neural_net,' done in',time.time()-start_time)
        return np.array(results_pred),np.array(results_id),np.array(results_mask)

    def postprocess(test,mean,std):
        test_n=test*std
        test_n=test_n+mean
        return test_n

    def mask(results,mask):
        final_prediction=results[:,:,:]*np.expand_dims(mask[:,:],axis=2)
        return final_prediction
    print('Preprocess of',neural_net)
    x_test_norm=preprocess(testx,parameters[neural_net]['meanx'],parameters[neural_net]['stdx'])
    print('Enter ',neural_net)
    results_pred,results_id,results_mask=predict(hyper_parameters,parameters[neural_net]['parameters'],x_test_norm,testm,testid,neural_net)
    predictions_y1=postprocess(results_pred,parameters[neural_net]['meany'],parameters[neural_net]['stdy'])
    predictions_y1=mask(predictions_y1,results_mask)
    return predictions_y1,results_mask,results_id

def join_predictions(pred1,pred2,pred3,pred4):
    final_prediction = (pred1+pred2+pred3+pred4)/4
    return final_prediction

def save_predictions(aa,x,pred,mask,id,output):
    np.savez('./predicted_proteins/'+str(output)+'.npz',id=id,aminoacid=aa,x=x,mask_seq=mask,predicted=pred)    

predictions_y1,mask1,id1=prediction_neuralnet(x,mask,id,parameters,hyper_parameters,'neural_net1')
predictions_y2,mask2,id2=prediction_neuralnet(x,mask,id,parameters,hyper_parameters,'neural_net2')
predictions_y3,mask3,id3=prediction_neuralnet(x,mask,id,parameters,hyper_parameters,'neural_net3')
predictions_y4,mask4,id4=prediction_neuralnet(x,mask,id,parameters,hyper_parameters,'neural_net4')

prediction_mean=join_predictions(predictions_y1,predictions_y2,predictions_y3,predictions_y4)
save_predictions(aminoacid,x,prediction_mean,np.expand_dims(mask4,axis=2),id,args.output_file)

