import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

location = 'tiffset/'
num_classes = 5

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
    img = np.pad(img, pad_width=npad, mode='constant',constant_values=0)

    return img

best_dist = np.inf;
worst_dist = -1;
location = 'tiffset/'
# the number of discrete classes


print('reading filenames from filenames.txt')
dt = np.dtype([('filename','|S20'),('labels',np.int32,(num_classes,))])
filedata = np.loadtxt(location + 'filenames.txt',dtype=dt)


IMG_HEIGHT = 178 # biggest picture in the set
IMG_WIDTH = 177




correct_count = 0;
total = 0;

for n in range(filedata['filename'].shape[0]):

    best_dist = np.inf;
    worst_dist = -1;

    infile = (filedata['filename'][n]).decode('UTF-8')
    correct_output = (filedata['labels'][n])
   
    
    in_img = mpimg.imread(location +infile)
    IN_HEIGHT = in_img.shape[0]
    IN_WIDTH = in_img.shape[1]
    in_img = pad_img(in_img,IMG_HEIGHT,IMG_WIDTH,IN_HEIGHT,IN_WIDTH)
    in_oneD = in_img.reshape(IMG_HEIGHT * IMG_WIDTH)
    
    for k in range(filedata['filename'].shape[0]):
        
        if(filedata['filename'][k].decode('UTF-8') == infile):
            # print(filedata['filename'][k].decode('UTF-8') + ' == ' + infile + '. skipping.')
            continue 
        
        cur_file = filedata['filename'][k].decode('UTF-8')
        cur_img = mpimg.imread('tiffset/'+cur_file)
        
        # print(cur_file)
    
        CUR_HEIGHT = cur_img.shape[0]
        CUR_WIDTH = cur_img.shape[1]
        cur_img = pad_img(cur_img,IMG_HEIGHT,IMG_WIDTH,CUR_HEIGHT,CUR_WIDTH)
        
        cur_oneD = cur_img.reshape(IMG_HEIGHT * IMG_WIDTH)
    
        temp_dist = 0;
    
        for i in range(in_oneD.shape[0]):
            
            temp_dist += np.power((int(in_oneD[i]) - int(cur_oneD[i])),2)
            
        if(temp_dist < best_dist):
            # print('new best distance found: ' + str(temp_dist))
            best_dist = np.copy(temp_dist)
            best_guess = filedata['filename'][k].decode('UTF-8')
            best_guess_labels = filedata['labels'][k]
        
        if(temp_dist > worst_dist):
            worst_dist = np.copy(temp_dist)    
            # print('new worst distance found: ' + str(temp_dist))
            
    total += 1
    if(np.array_equal(correct_output,best_guess_labels)):
        correct_count += 1
    
    
    print('Input file: '+ infile + ' | labels: ' + str(correct_output))
    print('Best guess: ' + best_guess + ' | labels: ' +  str(best_guess_labels))
    print(str(correct_count)+ '/' + str(total) + ' correct')








