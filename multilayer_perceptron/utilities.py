import numpy as np
import theano.tensor as T
from collections import OrderedDict
import matplotlib.image as mpimg
def progress_bar(current,total):
    div = current/total
    bar_length = 20
    percentage = 100*div
    progress = '#'*int(bar_length*div) + ' '*(bar_length-int(bar_length*div))
    out_str = "Progress: [%s] %.3f%%\r" % (progress,percentage)
    sys.stdout.write(out_str)
    sys.stdout.flush()
    if(current >= total):
        print("Complete.")
    

def pinv(M,eps=0.0001):
    u,s,v = np.linalg.svd(M)
    u = u.transpose()
    v = v.transpose()
    S = np.zeros((v.shape[0],u.shape[0]))
    S[:len(s),:len(s)] = np.diag(1/s) + eps*np.identity(len(s))
    return np.dot(v,np.dot(S,u))

def normalise(A):
    return (A - A.mean(axis=0))/(A.std(axis=0))

def softmax(h):
    h = np.array(h)
    if(len(h.shape) == 2):
        return np.array([np.exp(x)/np.sum(np.exp(x), axis=0) for x in h])
    elif(len(h.shape) == 1):
        return np.exp(h)/np.sum(np.exp(h),axis=0)
    else:
        return h

def unit(a):
    return np.divide(a,np.sum(a))
    
def one_hot(index,size):
    lst = [0]*size
    lst[index] = 1
    return lst


def grad_desc(cost, theta):
    return theta - (learning_rate*(1) * T.grad(cost, wrt=theta))
    
    
def pad_img(img,IMG_HEIGHT,IMG_WIDTH,IN_HEIGHT,IN_WIDTH):    
    vpad = np.abs((IMG_HEIGHT - IN_HEIGHT)/2)
    if not (np.floor(vpad) ==  vpad):
        tpad = np.floor(vpad)
        bpad = (vpad + 1)
    else:
        tpad = vpad
        bpad = vpad    
    hpad = np.abs((IMG_WIDTH - IN_WIDTH)/2)
    if not (np.floor(hpad) == hpad):
        lpad = np.floor(hpad)
        rpad = (hpad + 1)
    else:
        lpad = hpad
        rpad = hpad        
    tpad = (int)(tpad)
    bpad = (int)(bpad)
    lpad = (int)(lpad)
    rpad = (int)(rpad)
    npad = ((tpad,bpad),(lpad,rpad))
    if(IN_HEIGHT > IMG_HEIGHT and IN_WIDTH > IMG_WIDTH):
        img = img[lpad: (IN_WIDTH - rpad), tpad: (IN_HEIGHT - bpad)]
    else:
        img = np.pad(img, pad_width=npad, mode='constant',constant_values=0)
    return img
    
    
def confusion_matrix(pred,target,num_classes):
    

    confusion_matrix = np.zeros((num_classes,num_classes))
    for guess in range(len(pred)):
        confusion_matrix[target[guess]][pred[guess]] += 1
    
    return confusion_matrix
    
    
def get_images(w,h,file_list=None,num_classes=5):
    
    infile = 'filenames5.txt'
    folder = 'tiffs5/'
    abspath = 'C:/Users/Roan Song/Desktop/thesis/'
    
    if(not file_list):
        dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_classes,))])
        infile = 'filenames5.txt'
        filedata = np.loadtxt(infile,dtype=dt)
        file_list = [a.decode('UTF-8') for a in filedata['filename']]
        file_list.sort(key=lambda x:x[-7:])
    
    suffixes = OrderedDict()
    
    for f in file_list:
        suffixes[f[-7:]] = suffixes.get(f[-7:], 0) + 1
    
    ind = 0
    for i in suffixes:    
        suffixes[i] = {"count":suffixes[i],"label":one_hot(ind,num_classes)}
        ind += 1    
    
    
    img_arr = np.zeros((len(file_list),h*w))
    target_arr = np.zeros((len(file_list),num_classes))
    i = 0
    for fname in file_list:
        img = mpimg.imread(abspath + folder + fname)
        IN_HEIGHT = img.shape[0]
        IN_WIDTH = img.shape[1]
        
        img = pad_img(img,h,w,IN_HEIGHT,IN_WIDTH)
        oneD = img.reshape(h * w)
        oneD = normalise(oneD)
        img_arr[i] = oneD  
        target_arr[i] = suffixes[fname[-7:]]["label"]
         
        i+=1 
    return img_arr,target_arr,suffixes