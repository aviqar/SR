import numpy as np

res_trans_newbrain = np.load('/Volumes/GPN2/Asim/data/3_res_trans_newbrain.npy')


a = 768
sc = 8

d = np.zeros((a/sc*a/sc,2))
count = 0
for i in range(0,a,sc):
    for j in range(0,a,sc):
        d[count,:] = [i,j]
        count = count + 1

newbrainhirez = np.zeros((a,a,121))
for k in range(121):
    im = np.ones((a,a))
    im = im * np.nan
    lorezbrain = res_trans_newbrain[:,:,k]
    lorezbrain = lorezbrain.reshape((9216,))
    im[np.int16(d)[:,0],np.int16(d)[:,1]] = lorezbrain
    newbrainhirez[:,:,k] = im

np.save('/Volumes/GPN2/Asim/data/4_re-upsample.npy',newbrainhirez)