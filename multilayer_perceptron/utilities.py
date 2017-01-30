import numpy as np
import theano.tensor as T
from collections import OrderedDict
import matplotlib.image as mpimg
import sys
import timeit
def progress_bar(current,total):
    """
    A method to display a progress bar that updates on the same line
    
    current --- the current value or state of progress
    total   --- the value at which progress will be 100%
    """
    
    div = current/total
    bar_length = 20
    percentage = 100*div
    progress = '#'*int(bar_length*div) + ' '*(bar_length-int(bar_length*div))
    out_str = "Progress: [%s] %.3f%%\r" % (progress,percentage)
    sys.stdout.write(out_str)
    sys.stdout.flush()
    if(current >= total):
        print("Complete.")
    

def normalise(A):
    """
    Normalises a vector so that most of its values lie between -1 and 1
    returns the vector mentioned above
    A --- the vector to be normalised
    """
    return (A - A.mean(axis=0))/(A.std(axis=0))

def unit(a):
    """
    Takes an array of numbers and scales them so that the sum of the array is 1
    returns the scaled array
    
    a --- the array to scale
    """
    return np.divide(a,np.sum(a))
    
def one_hot(index,size):
    """
    Creates a one-hot vector from a number.
    returns a one-hot vector
    
    index --- the index of the vector to set to 1
    size  --- the size of the vector
    
    e.g. if index = 2 and size = 4:
    return [0, 0, 1, 0]
    """
     
    lst = [0]*size
    lst[index] = 1
    return lst


    
def pad_img(img,IMG_HEIGHT,IMG_WIDTH,IN_HEIGHT,IN_WIDTH):  
    """
    Resizes an image to the desired size in pixels
    If the new size is smaller, the image is cropped
    If the new size is larger, the image is centered and padded with zeros
    returns the processed image
    
    img        --- the image to be processed
    IMG_HEIGHT --- the desired height of the image
    IMG_WIDTH  --- the desired width of the image
    IN_HEIGHT  --- the height of the input image
    IN_WIDTH   --- the width of the input image
    
    """  
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
    
    
def confusion_matrix(pred,target,num_classes,percent=True):
    """
    A method to create a confusion matrix based on two arrays of predicted classes
    returns the confusion matrix
    
    pred        --- array of predicted values
    target      --- array of correct values
    num_classes --- number of classes and the shape of the confusion matrix
    percent     --- uses percent values in the confusion matrix instead of raw numbers (default: True)
    """

    confusion_matrix = np.zeros((num_classes,num_classes))
    for guess in range(len(pred)):
        confusion_matrix[target[guess]][pred[guess]] += 1
    
    if(percent):
        for row in range(num_classes):
            confusion_matrix[row] = 100.*confusion_matrix[row]/confusion_matrix[row].sum()
    
    return confusion_matrix
    
    
def get_images(w,h,file_list=None,num_classes=8,threshold=False,noise=False):
    """
    Load images from a file, crop/pad them to a specific size, with some pre-processing options
    returns an array of image vectors, an array of target vectors, and 
    a dictionary linking the suffixes of each file to their respective target vector and 
    the count of each class within the dataset.
    
    
    w           --- desired width of the images
    h           --- desire height of the images
    file_list   --- if specified, list of files to read image data from, otherwise uses default (default: None)
    num_classes --- number of classes into which images can be classified
    threshold   --- if True, set values under the median of each image to zero (default: False)
    noise       --- if True, add Gaussian noise to each image (default: False)
    """
    infile = str(num_classes) +'.txt'
    folder = 'tiffs'+str(num_classes)+'/'
    abspath = 'C:/Users/Roan Song/Desktop/thesis/'
    rng = np.random.RandomState(0)
    
    
    if(not file_list):
        dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_classes,))])
        infile = 'filenames8.txt'
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
        image = img.reshape(h * w)
        
        if(threshold):
            below_thresh = image < np.mean(image)
            image[below_thresh] = 0
        
        image = normalise(image)
        
        if(noise):
            image += rng.normal(0,1,image.shape)
        
        img_arr[i] = image  
        target_arr[i] = suffixes[fname[-7:]]["label"]
         
        i+=1 

    return img_arr,target_arr,suffixes
    
def gen_sets(data,targets,train,val,test):
    """
    Generate training, validation and test subsets from a given dataset.
    Returns the three sets and the indices of the original dataset which correspond to them
    
    data    --- full dataset to be split
    targets --- targets component of the dataset
    train   --- proportion allocated to the training set
    val     --- proportion allocated to the validation set
    test    --- proportion allocated to the test set
    
    Note: train, val and test do not have to sum to 1. 
    The unit function is applied to them, ensuring that they sum to 1.
    """
    
    train,val,test = unit([train,val,test])
    
    training_set   = np.zeros((int(len(data)*train),2))
    validation_set = np.zeros((int(len(data)*val  ),2))
    test_set       = np.zeros((int(len(data)*test ),2))
    rng = np.random.RandomState(0)
    indices = np.arange(len(data))
    
    temp = rng.choice(indices,size=len(training_set),replace=False)
    training_indices = temp
    training_set = (np.vstack(data[temp]),np.vstack(targets[temp]))
    # training_set = (data[temp],targets[temp])
    indices = np.delete(indices,temp)
    
    temp = rng.choice(indices,size=len(validation_set),replace=False)
    validation_indices = temp
    validation_set = (np.vstack(data[temp]),np.vstack(targets[temp]))
    indices = np.delete(indices,temp)
    
    temp = rng.choice(indices,size=len(test_set),replace=False)
    testing_indices = temp
    test_set = (np.vstack(data[temp]),np.vstack(targets[temp]))
    indices = np.delete(indices,temp)
    
    return training_set, validation_set, test_set, (training_indices,validation_indices,testing_indices)

    
    # forfiles /m *.026 /c "cmd /c mstar2tiff -i @file -o tiffs/@file.tif -e"