import scipy.io

data = scipy.io.loadmat('/Volumes/GPN2/HBI_350021_swinorm_33.mat')
newbrain = data['img']
newbrain = newbrain[:,128:1024-128]

np.save('/Volumes/GPN2/Asim/data/1_newbrain.npy',newbrain)
