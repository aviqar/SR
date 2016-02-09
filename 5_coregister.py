import numpy as np
from skimage import transform as tf

newbrainhirez = np.load('/Volumes/GPN2/Asim/data/4_re-upsample.npy')

x = np.arange(-5,5.2,1)
y = np.arange(-5,5.2,1)
translations = np.zeros((121,2))
count = 0
for i in x:
    for j in y:
        translations[count,:] = [i,j]
        count = count + 1

redo_translations = translations * -1

count = 0
coreg_newbrain = np.zeros([newbrainhirez.shape[0],newbrainhirez.shape[1],newbrainhirez.shape[2]])
for trans in redo_translations:
    tform = tf.SimilarityTransform(scale = 1, rotation = 0, translation = trans)
    coreg_newbrain[:,:,count] = tf.warp(newbrainhirez[:,:,count], tform)
    count = count + 1
    
np.save('/Volumes/GPN2/Asim/data/5_coreg_newbrain.npy',coreg_newbrain)