import scipy.io
import numpy as np

data = scipy.io.loadmat('/Volumes/GPN2/SR/HBI_350021_swinorm_33.mat')
newbrain = data['img']
newbrain = newbrain[:,128:1024-128]

np.save('/Volumes/GPN2/SR/Nearest Neighbor/data/originalImage.npy', newbrain)
