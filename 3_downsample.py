import numpy as np
import scipy.misc

trans_newbrain = np.load('/Volumes/GPN2/Asim/data/2_trans_newbrain.npy')

count = 0
res_trans_newbrain = np.zeros([trans_newbrain.shape[0]/8,trans_newbrain.shape[1]/8,trans_newbrain.shape[2]])
for i in range(121):
    res_trans_newbrain[:,:,count] = scipy.misc.imresize(trans_newbrain[:,:,count],(96,96))
    count = count + 1
    
np.save('/Volumes/GPN2/Asim/data/3_res_trans_newbrain.npy',res_trans_newbrain)

//ugh