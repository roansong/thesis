import matplotlib.pyplot as plt
# import matplotlib.rcsetup as rcsetup
import matplotlib.image as mpimg
import numpy as np


def unitise(arr):
    out_arr = (arr - np.mean(arr))/np.std(arr)
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
    
    img = np.pad(img, pad_width=npad, mode='constant',constant_values=0)
    return img



#### figure out the image dimensions
#### I scale up the smaller images

IMG_HEIGHT = 193
IMG_WIDTH = 192
num_labels = 2
num_images = 10
img_arr = np.zeros((num_images,IMG_HEIGHT*IMG_WIDTH))


# dt = 'S16,i4,i4'
dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_labels,))])
filedata = np.loadtxt('filedata.txt',dtype=dt)
print(filedata)

correct_guesses = 0
learning_rate = 0.5;

###### initialise weights and layer structure
layer1_size = 10
layer2_size = 10
layer1 = np.zeros(layer1_size)
l1_net = np.zeros(layer1_size)
l1_activation =  np.zeros(layer1_size)
layer2 = np.zeros(layer2_size)
l2_net = np.zeros(layer2_size)
l2_activation =  np.zeros(layer2_size)
pred = np.zeros(num_labels)
pred_net = np.zeros(num_labels)
output = np.zeros(num_labels)

# creating a weights vector (number of pixels, output number
in_weights = np.zeros((IMG_HEIGHT*IMG_WIDTH,layer1_size))

# create random weights for each pixel (neuron)
for i in range(IMG_HEIGHT*IMG_WIDTH):
    in_weights[i] = np.random.uniform(-0.5, 0.5, size = layer1_size)


# create random weights for each neuron in the first layer
l1_weights = np.zeros((layer1_size,layer2_size))
for i in range(layer1_size):
    l1_weights[i] = np.random.uniform(-0.5, 0.5, size = layer2_size)




# create random weights for each neuron in the second layer
l2_weights = np.zeros((layer2_size,num_labels))
for i in range(layer2_size):
    l2_weights[i] = np.random.uniform(-0.5, 0.5, size = num_labels)

for n in range(num_images):
    
    print()
    print('Image ' + str(n + 1))
    # read in image data
    fname = filedata['filename'][n].decode('UTF-8')
    img = mpimg.imread('tiffs/'+fname)
    IN_HEIGHT = img.shape[0]
    IN_WIDTH = img.shape[1]
    
    #read labels
    correct_output = filedata['labels'][n]
    
    # pad image
    img = pad_img(img,IMG_HEIGHT,IMG_WIDTH,IN_HEIGHT,IN_WIDTH)

    # reshape and unitise image data
    oneD = img.reshape(IMG_HEIGHT * IMG_WIDTH)
    oneD = unitise(oneD)
    img_arr[n] = oneD

    
    ################################################
    ## INPUT LAYER COMPLETE
    
    # forfiles /p DIR /m *.0* /c "cmd /c C:\Users\roans\Desktop\thesis\MATLAB\mstar2tiff -i @file -o @file.tiff -e"
    #### 1st level of weights
    
    
    layer1 = np.zeros(layer1_size)
    
    
    # for each pixel
    # add the weighted value to each neuron in the first layer
    for i in range(img_arr[n].shape[0]):
        for j in range(layer1.shape[0]):
            layer1[j] += img_arr[n][i]*in_weights[i][j]
        
    l1_net = np.copy(layer1)
    # for each neuron in the first layer   
    # calculate the activation 
    for i in range(layer1.shape[0]):
        layer1[i] = np.tanh(layer1[i])
    
    l1_activation = np.copy(layer1)
    
    # print(layer1)
    print("1st layer activations calculated")
    
    #########
    
    
    
    
    
    # for each neuron in the first layer
    # add the weighted value to each neuron in the second layer
    for i in range(layer1_size):
        for j in range(layer2_size):
            layer2[j] += layer1[i]*l1_weights[i][j]
    l2_net = np.copy(layer2)
    
    # for each neuron in the second layer   
    # calculate the activation 
    for i in range(layer2_size):
        layer2[i] = np.tanh(layer2[i])
    l2_activation = np.copy(layer2)
    
    # print(layer2)
    print("2nd layer activations calculated")
    
    #########
    
    
    
    
    
    
    # for each neuron in the second layer
    # add the weighted value to each neuron in the output
    for i in range(layer2_size):
        for j in range(pred.shape[0]):
            pred[j] += layer2[i]*l2_weights[i][j]
    pred_net = np.copy(pred)
    
    # for each neuron in the output layer   
    # calculate the activation 
    
    for i in range(pred.shape[0]):
        pred[i] = np.tanh(pred[i])
    output = np.copy(pred)
    
    # print(output)
    print("output layer activations calculated")
    print(str(output) + ' vs ' + str(correct_output))
    # print("apply softmax?")
    
    

    total_error = 0
    error = np.zeros(num_labels)
    for i in range(num_labels):
        error[i] = 0.5*np.power(correct_output[i] - output[i],2)
    total_error = np.sum(error)
    
    
    #start by figuring out change for outermost weights
    
    errvsout = -(correct_output[0] - output[0])
    actderiv = 1 - np.power(np.tanh(pred_net[0]),2)
    l2_out = l2_activation[0]
    
    errvsweight = errvsout*actderiv*l2_out
    # weight being changed, but we use the original weights during the rest of the backpropagation
    new_l3_weights = np.copy(pred_net)
    new_l3_weights[0] = pred_net[0] - learning_rate*errvsweight
    
    
    #derivative of tanh(x) is 1 - tanh(x)^2
    
    print('total error: '+str(total_error))
    
    
    # after out layer is done, we move a layer in
    # remember only changing weights afterwards
    
    break
    
    
    
    
    
    
    
    
    

    ####################
    # error calculations
    # backpropagation!



print("done!")





# print(img.shape)
# print(b.shape)
# imgplot = plt.imshow(img,cmap='gray',interpolation='none')
# plt.show()
# plt.set_cmap('hot')
# plt.colorbar

# print(rcsetup.all_backends)
# import matplotlib
# print(matplotlib.matplotlib_fname())