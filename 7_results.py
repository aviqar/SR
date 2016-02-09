import numpy as np
import matplotlib.pyplot as plt
#import results

mean = np.load('/Volumes/GPN2/Asim/data/7_newbrainmean.npy')
newbrain_auto = np.load('/Volumes/GPN2/Asim/data/6_newbrain_auto.npy')
newbrain = np.load('/Volumes/GPN2/Asim/data/1_newbrain.npy')

#compare results to original image
meandiff = newbrain - mean
SRdiff = newbrain - newbrain_auto


np.save('/Volumes/GPN2/Asim/data/8_meandiff.npy',meandiff)
np.save('/Volumes/GPN2/Asim/data/9_SRdiff.npy',SRdiff)

meandiff = np.load('/Volumes/GPN2/Asim/data/8_meandiff.npy')
SRdiff = np.load('/Volumes/GPN2/Asim/data/9_SRdiff.npy')

#display images

plt.figure()
plt.imshow(meandiff)
plt.colorbar()
plt.clim(-500,500)

plt.figure()
plt.imshow(SRdiff)
plt.colorbar()
plt.clim(-500,500)