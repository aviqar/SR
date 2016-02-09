import numpy as np
resamplednewbrain = np.load('/Volumes/GPN2/Asim/data/5_coreg_newbrain.npy')
import scipy.signal
newbrain = np.load('/Volumes/GPN2/Asim/data/1_newbrain.npy')
 
#load function to find 5x5 neighbors

def find_neighbors(m, i, j, dist):    
    neighbors = []    
    i_min = max(0, i-dist) 
    i_max = i+dist+1    
    j_min = max(0, j-dist)    
    j_max = j+dist+1    
    for row in m[i_min:i_max]:        
        neighbors.append(row[j_min:j_max])
    neighbors = np.array(neighbors)
    return neighbors
    
def corr(x,y):
    cc = scipy.signal.correlate(x,y,'same')
    cc = cc/np.max(cc)
    return cc

N = np.zeros((5,5,121))
M = np.zeros((5,5,121))
SR_rerun = np.zeros((768,768))
for x in range(2,765):
    for y in range(2,765):
       
        for i in range(121):
            N[:,:,i] = find_neighbors(resamplednewbrain[:,:,i],x,y,2)
            M[:,:,i] = find_neighbors(newbrain,x,y,2)
        g = np.nanmean(N,2) 
        d = np.nanmean(M,2)
        R = np.linalg.pinv(corr(g,g))
        P = corr(g,d)
        W = np.dot(R,P)
        SR_rerun[x,y] = np.mean(np.dot(W.T,g))
        
np.save('/Volumes/GPN2/Asim/data/10_SR_rerun.npy',SR_rerun)

newbrain_auto = np.load('/Volumes/GPN2/Asim/data/6_newbrain_auto.npy')
Z = (newbrain_auto - np.nanmean(np.reshape(newbrain_auto,(1,768*768)))) / np.nanstd(np.reshape(newbrain_auto,(1,768*768)))
newbrain_new = newbrain_auto
newbrain_new[abs(Z)>1] = np.nan



