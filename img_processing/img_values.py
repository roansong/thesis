import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def unitise(arr):
    out_arr = (arr - np.mean(arr))/np.std(arr)
    return out_arr

def clamp(arr,threshold):    
    out_arr = np.zeros(arr.shape[0])    
    for i in range(arr.shape[0]):
        if(arr[i] > threshold):
            out_arr[i] = arr[i]
        else:
            out_arr[i] = 0            
    return out_arr

def pad_img(img,IMG_HEIGHT,IMG_WIDTH,IN_HEIGHT,IN_WIDTH):
    
    vpad = (IMG_HEIGHT - IN_HEIGHT)/2
    if not (np.floor(vpad) ==  vpad):
        tpad = np.floor(vpad)
        bpad = (vpad + 1)
    else:
        tpad = vpad
        bpad = vpad
    
    hpad = (IMG_WIDTH - IN_WIDTH)/2
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
 
 
    
    img = np.pad(img, pad_width=npad, \
                 mode='constant',constant_values=0)
    return img

IMG_HEIGHT = 193
IMG_WIDTH = 192
num_labels = 2
num_images = 10
img_arr = np.zeros((num_images,IMG_HEIGHT*IMG_WIDTH))


dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_labels,))])
filedata = np.loadtxt('filedata.txt',dtype=dt)
# print(filedata)


fig = plt.figure()


targets = 5
thresholds = 5

# for n in range(filedata['filename'].shape[0]):
for n in range(targets):
    fname = filedata['filename'][n]
    fname = fname.decode('UTF-8')
    img = mpimg.imread('tiffs/'+fname)
    IN_HEIGHT = img.shape[0]
    IN_WIDTH = img.shape[1]
    img = pad_img(img,IMG_HEIGHT,IMG_WIDTH,IN_HEIGHT,IN_WIDTH)
    img = unitise(img)
    oneD = img.reshape(IMG_HEIGHT * IMG_WIDTH)
    
    for i in range(1, thresholds + 1):
        title_text = fname
        
        temp_img = clamp(oneD,i).reshape(IMG_HEIGHT,IMG_WIDTH)
        plt.subplot(targets,thresholds,n*targets + i)
        imgplot = plt.imshow(temp_img,cmap='gray',interpolation='none')
    
        title_text += " clamped below " + str(i)
        plt.title(title_text)
        plt.show()


plt.suptitle('Target chips with values below a certain threshold set to zero')
# 
# oneD = clamp(oneD,5)
# print(fname + " " +  str(np.mean(img)))
# print(fname + " " +  str(np.mean(oneD)))



# plt.subplot(2,1,1)
# imgplot = plt.imshow(img,cmap='gray',interpolation='none')
# plt.show()
# plt.colorbar()
# plt.subplot(2,1,2)
# imgplot = plt.imshow(oneD.reshape(IMG_HEIGHT,IMG_WIDTH),cmap='gray',interpolation='none')
# plt.show()
# # plt.set_cmap('hot')
# plt.colorbar()