import matplotlib.pyplot as plt
import time
# import matplotlib.rcsetup as rcsetup
import matplotlib.image as mpimg
import numpy as np
np.random.seed(1)
np.set_printoptions(threshold=1000)
np.set_printoptions(precision=6) # default 8
def unitise(arr):
    out_arr = (arr - np.mean(arr))/np.std(arr)
    return out_arr

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

def crop_img(img,IMG_HEIGHT,IMG_WIDTH,IN_HEIGHT,IN_WIDTH):
    
    vpad = (IN_HEIGHT - IMG_HEIGHT)/2
    if not (np.floor(vpad) ==  vpad):
        tpad = np.floor(vpad)
        bpad = (vpad + 1)
    else:
        tpad = vpad
        bpad = vpad
    
    hpad = (IN_WIDTH - IMG_WIDTH)/2
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
    
    img = img[lpad: (IN_WIDTH - rpad), tpad: (IN_HEIGHT - bpad)]
    

    return img

#### figure out the image dimensions
#### I scale up the smaller images

IMG_HEIGHT = 40
IMG_WIDTH = 40
num_labels = 5
num_images = 70
img_arr = np.zeros((num_images,IMG_HEIGHT*IMG_WIDTH))

infile = 'filedata.txt'
infile = 'filenames.txt'
# dt = 'S16,i4,i4'
dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_labels,))])
filedata = np.loadtxt(infile,dtype=dt)
filedata['labels'][filedata['labels'] == 0] = -1

correct_guesses = 0


###### initialise weights and layer structure
layer1_size = 50
layer2_size = 50
layer1 = np.zeros(layer1_size)
l1_net = np.zeros(layer1_size)
l1_out =  np.zeros(layer1_size)
layer2 = np.zeros(layer2_size)
l2_net = np.zeros(layer2_size)
l2_out =  np.zeros(layer2_size)
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

#34/70 at 267 bias = 0.1 and ceiling thing is on max 34
#30/70 at 267 bias = 0 and ceiling thing is off error 80
#34/70 at 267 bias = 0.1 ceiling thing is off max 35 error 78
start = time.clock()
# print(start)
bias = 0.0
bias1 = bias
bias2 = bias
rand_rate = True
learning_rate = 0.2
if(rand_rate):
    learning_rate = np.random.uniform(0, 0.25)
iterations = 10000
epochs = 20
generations = 1
epochs = (int)(iterations/generations)
print("Number of images: " + str(num_images))
print("Number of classes: " + str(num_labels))
print("Image size: " + str(IMG_WIDTH) + "x" + str(IMG_HEIGHT))
print("Iterations: " + str(iterations))
print("Epochs: " + str(epochs))
print("Layer 1 size: " + str(layer1_size))
print("Layer 2 size: " + str(layer2_size))
print("Neuron bias: " + str(bias))
print("Back-propagation depth: 2 layers")
print("Learning rate: " + str(learning_rate))
results_arr = np.zeros((epochs,generations,num_images))
epoch_err = np.zeros(epochs)
for epoch in range(epochs):
    # print("Epoch " + str(epoch))
    for gen in range(generations):
        correct_guesses = 0
        correct_classes = np.zeros(num_labels)
        # print("Generation " + str(gen) + " (epoch " + str(epoch) + ")")
        for n in range(num_images):
        
            # print()
            # if(n == 0):
            #     print('Image ' + str(n + 1))
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
            
            # forfiles /p DIR /m *.0* /c "cmd /c C:\Users\Roan Song\Desktop\thesis\MATLAB\mstar2tiff -i @file -o @file.tiff -e"
            #### 1st level of weights
            
            
            layer1 = np.zeros(layer1_size)
            
            
            # for each pixel
            # add the weighted value to each neuron in the first layer
            for i in range(img_arr[n].shape[0]):
                for j in range(layer1_size):
                    layer1[j] += img_arr[n][i]*in_weights[i][j]
            
            #add bias (usually zero)
            layer1 += bias1
                
            l1_net = np.copy(layer1)
            # for each neuron in the first layer   
            # calculate the activation 
            for i in range(layer1.shape[0]):
                layer1[i] = np.tanh(layer1[i])
            
            l1_out = np.copy(layer1)
            
            # print(layer1)
            # print("1st layer activations calculated")
            
            #########
            
            
            
            
            
            # for each neuron in the first layer
            # add the weighted value to each neuron in the second layer
            for i in range(layer1_size):
                for j in range(layer2_size):
                    layer2[j] += layer1[i]*l1_weights[i][j]
            
            #add bias (usually zero)        
            layer2 += bias2
            
            
            l2_net = np.copy(layer2)
            
            # for each neuron in the second layer   
            # calculate the activation 
            for i in range(layer2_size):
                layer2[i] = np.tanh(layer2[i])
            l2_out = np.copy(layer2)
            
            # print(layer2)
            # print("2nd layer activations calculated")
            
            #########forfiles /m *.0* /s /c "cmd /c C:\TEST\mstar2tiff.exe -i BTR_60/@file -o test.tif" 

            
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
            
            
            if(np.argmax(output) == np.argmax(correct_output)):
                correct_guesses += 1
                correct_classes[np.argmax(correct_output) - 1] += 1
            
            #if(output[np.argmax(output) -1] >= 0.75):
            #    output[np.argmax(output) -1] = 1
            
            
            #big ol' breakthrough right here
            # the best guess is set to one, e.g. [0 0.5 0.3] becomes [0 1 0.3]
            # this rewards correct guesses by lessening the error
            # it increases the penalty for incorrect guesses
            
            
            # print(output)
            # print("output layer activations calculated")
            
            # if(gen == 0 and n == 0):
            #     print(str(output) + ' vs ' + str(correct_output))
           
            
            
        
            err_total = 0
            error = np.zeros(num_labels)
            for i in range(num_labels):
                error[i] = 0.5*np.power(correct_output[i] - output[i],2)
            err_total = np.sum(error)
            
            # if(n ==0):
            #     print("Error: " + str(err_total))
            
            
            l2_weights_new = np.copy(l2_weights)
            
            # l2_weights.shape[0] = number of neurons in layer 2
            # l2_weights.shape[1] = number of output neurons
            
            # for l2_neuron in range(layer2_size): 
            #     for l2_weight in range(output.size):
            #         p1 = -(output[l2_weight] - correct_output[l2_weight])
            #         p2 = (1 - np.power(np.tanh(pred_net[l2_weight]),2))
            #         p3 = l2_weights[l2_neuron][l2_weight]
            #         p4 = l2_out[l2_neuron]
            #         
            #         change = -(output[l2_weight] - correct_output[l2_weight]) * (1 - np.power(np.tanh(pred_net[l2_weight]),2)) * l2_out[l2_neuron]
            #         
            #         l2_change = p1*p2*p4
            #         l2_weights_new[l2_neuron][l2_weight] = l2_weights[l2_neuron][l2_weight] + learning_rate*change
            
            
            l1_weights_new = np.copy(l1_weights)
            
            # l2_net
            # l2_out
            # l1_net
            # l1_out
            
            
            l1_weights_change = np.zeros(l1_weights.shape[0])
            
            # d_err = d_err1/d_l1_out[0] + d_err2/d_l1_out[0] + d_err3/d_l1_out[0] etc
            
            
            # for neuron in range(l1_weights.shape[0]):
                # d_err = d_err1/d_l1_out[neuron] + d_err2/d_l1_out[neuron] + d_err3/d_l1_out[neuron] etc
                
                # d_err = d_err1/d_l1_out[neuron]
                
            # Eo1/outh1 = -(output[0] - correct_output[0]) * (1 - np.power(np.tanh(pred_net[0]),2)) * l2_weights[0][0]
            # 
            # Eo2/outh1 = -(output[1] - correct_output[1]) * (1 - np.power(np.tanh(pred_net[1]),2)) * l2_weights[0][1]
            # 
            # Eo3/outh1 = -(output[2] - correct_output[2]) * (1 - np.power(np.tanh(pred_net[2]),2)) * l2_weights[0][2]
            # 
            # sum above
            
            for l1_neuron in range(layer1_size):
            # l2_neuron ===== l1_weight
                for l2_neuron in range(layer2_size):
                    err_temp = 0
                    for l2_weight in range(output.size):
                        
                        p1 = -(output[l2_weight] - correct_output[l2_weight])
                        p2 = (1 - np.power(np.tanh(pred_net[l2_weight]),2))
                        p3 = l2_weights[l2_neuron][l2_weight]
                        p4 = l2_out[l2_neuron]
                        
                        l2_change = p1*p2*p4
                        l2_weights_new[l2_neuron][l2_weight] = l2_weights[l2_neuron][l2_weight] + learning_rate*l2_change
                        err_temp += p1*p2*p3
                    
                        # voila! you get the dEtotal/dl2_out[neuron]
        
                    l1_change = err_temp*(1 - np.power(np.tanh(l2_net[l2_neuron]),2)) * l1_out[l2_neuron]
                    l1_weights_new[l1_neuron][l2_neuron] = l1_weights[l1_neuron][l2_neuron] + learning_rate*l1_change
            
            # d_l2_out[l2_neuron]/dl2_net[l2_neuron] = (1 - np.power(np.tanh(l2_net[l2_neuron]),2))
            # d_l2_net[l2_neuron]/dl1_weights[l2_neuron] = l1_out[l2_neuron] 
            #     
            #     
            #     
            #     d = (1 - np.power(np.tanh(l2_net[0]),2)) * l1_out[l2_neuron]
                
                # neuron = error output number
                
                
                
                #tot/outo1 = -(output[weight] - correct_output[weight])
                #outo1/neto1 = (1 - np.power(np.tanh(pred_net[weight]),2))
                #neto1/w5 = l2_out[neuron]
                
                

                    
            
            
            
            
            
            
            
            
            l2_weights = np.copy(l2_weights_new)
            l1_weights = np.copy(l1_weights_new)
            
            results_arr[epoch][gen][n] = err_total

            
            
            # #derivative of tanh(x) is 1 - tanh(x)^2
            # 
            # print('total error: '+str(total_error))
            # 
            
            # after out layer is done, we move a layer in
            # remember only changing weights afterwards
            
            
    
    # print("Correct: " + str(correct_guesses) + "/" + str(num_images))        
    sum_of_err = 0
    for e in range(num_images):
        sum_of_err += np.abs(results_arr[epoch][-1][e])
    epoch_err[epoch] = sum_of_err
    
    if(rand_rate):4
        learning_rate = np.random.uniform(0, 0.25)
    
    
    print("Epoch " + str(epoch) + ": " + str(correct_guesses) + "/" + str(num_images) + " correct." + " " + str(learning_rate) +  " " +  str(correct_classes) + " " + str(sum_of_err))
print(epoch_err)        
print("Done!")
end = time.clock()
print("Time elapsed: " + str(end - start))
    

