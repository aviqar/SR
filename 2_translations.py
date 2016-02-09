import numpy as np
from skimage import transform as tf

newbrain = np.load('/Volumes/GPN2/Asim/data/1_newbrain.npy')


x = np.arange(-5,5.2,1)
y = np.arange(-5,5.2,1)

translations = np.zeros((121,2))
count = 0
for i in x:
    for j in y:
        translations[count,:] = [i,j]
        count = count + 1

count = 0
trans_newbrain = np.zeros((newbrain.shape[0],newbrain.shape[1],translations.shape[0]))
for trans in translations:   
    tform = tf.SimilarityTransform(scale=1, rotation=0, translation= trans)
    trans_newbrain[:,:,count] = tf.warp(newbrain, tform)
    count = count + 1
    
np.save('/Volumes/GPN2/Asim/data/2_trans_newbrain.npy', trans_newbrain)

