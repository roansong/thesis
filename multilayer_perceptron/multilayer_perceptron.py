import matplotlib.pyplot as plt
import time
# import matplotlib.rcsetup as rcsetup
import matplotlib.image as mpimg
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from random import random
np.random.seed(0)
np.set_printoptions(threshold=1000)
np.set_printoptions(precision=6) # default 8
from scipy.interpolate import spline


IMG_HEIGHT = 100
IMG_WIDTH = 100
num_labels = 5
num_images = 1291

#2/469
#3/743
#4/1017
#5/1291

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





infile = 'filenames5.txt'
folder = 'tiffs5/'
abspath = 'C:/Users/Roan Song/Desktop/thesis/'

# dt = 'S16,i4,i4'
dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_labels,))])
filedata = np.loadtxt(infile,dtype=dt)
filedata['labels'][filedata['labels'] == 0] = 0 # set to 0 if not using tanh

# filedata = np.random.permutation(filedata)

img_arr = np.zeros((num_images,IMG_HEIGHT*IMG_WIDTH))
labels_arr = np.copy(filedata['labels'])


for n in range(num_images):
    

    fname = filedata['filename'][n].decode('UTF-8')
    img = mpimg.imread(abspath + folder + fname)
    IN_HEIGHT = img.shape[0]
    IN_WIDTH = img.shape[1]
    
    img = pad_img(img,IMG_HEIGHT,IMG_WIDTH,IN_HEIGHT,IN_WIDTH)
    oneD = img.reshape(IMG_HEIGHT * IMG_WIDTH)
    # oneD = unitise(oneD)
    img_arr[n] = oneD






# THEANO TIME BOIZ



# theano.config.optmizer=None


def layer(X,w,bias=1):
    # b = np.array([bias], dtype=theano.config.floatX)
    # new_X = T.concatenate([X,b])
    # m = T.dot(w, new_X)
    m = T.dot(w.T, X)
    m += bias
    output = T.nnet.sigmoid(m)
    # output = np.tanh(m)
    return output


learning_rate = 0.1

def grad_desc(cost, theta):
    
    return theta - (learning_rate * T.grad(cost, wrt=theta))


def interleave(arr,classes):
    out_arr = np.copy(arr)
    
    divs = np.zeros(classes)
    for i in range(len(divs)):
        divs[i] = 0 + int(i*arr.shape[0]/classes)
    
    for i in range(int(arr.shape[0]/3)):
        for j in range(divs.size):
            out_arr[divs[j] + i] = arr[i*j + j] 
    
    return out_arr




x = T.dvector('x')
y = T.dvector('y')
output = T.dvector('output')

layer1_size = 30
layer2_size = 20

num_pixels = IMG_HEIGHT*IMG_WIDTH


l1_weights  = theano.shared(np.array(np.random.rand(num_pixels,layer1_size) , dtype=theano.config.floatX))
l2_weights  = theano.shared(np.array(np.random.rand(layer1_size,layer2_size), dtype=theano.config.floatX))
out_weights = theano.shared(np.array(np.random.rand(layer2_size,num_labels) , dtype=theano.config.floatX))
pred        = theano.shared(np.array(np.zeros(num_labels), dtype=theano.config.floatX))


show_weights = theano.function([],[l1_weights, l1_weights, out_weights])

def export_weights():
    
    l1, l2, out = show_weights()
    
    l1  = np.array(l1)[0]
    l2  = np.array(l2)[0]
    out = np.array(out)[0]
    
    return l1, l2, out
    





hidden_layer1 = layer(x, l1_weights)
hidden_layer2 = layer(hidden_layer1, l2_weights)
output =        layer(hidden_layer2, out_weights)
cst =           ((output - y)**2).sum()



# cost = theano.function(inputs = [x,y], 
#                        outputs=[cst,output], 
#                        updates=[
#                        (l1_weights, grad_desc(cst, l1_weights)),
#                        (l2_weights, grad_desc(cst, l2_weights)),
#                        (out_weights, grad_desc(cst, out_weights))])
                       
                       
                       
cost = theano.function(inputs = [x,y], outputs=[cst,output])
                       
update_weights = theano.function(inputs = [], 
                       outputs=[], 
                       updates=[
                       (l1_weights, grad_desc(cst, l1_weights)),
                       (l2_weights, grad_desc(cst, l2_weights)),
                       (out_weights, grad_desc(cst, out_weights))])



run = theano.function(inputs=[x],outputs=[output])


p = T.dscalar('p')
q = T.dscalar('q')

mn = p/q

my_mean = theano.function(inputs=[p,q],outputs=[mn])



# TRAINING

inputs = np.copy(img_arr)
# inputs = inputs[0:5]

y_pred = filedata['labels']
# y_pred = filedata['labels'][0:5]

cur_cost = 0
cost_arr = []
iterations = 301

start = time.clock()
cur_cost = np.zeros(len(inputs),dtype=theano.config.floatX)

output_arr = np.zeros((iterations,num_images,num_labels))
confusion_total = np.array(np.zeros((iterations,num_labels,num_labels)),dtype='int32')
confusion_matrix = np.zeros((num_labels,num_labels))


sample_size = 10
learning_rate = 0.01




for i in range(iterations):
    correct = 0
    correct_labels = np.zeros(num_labels)
    confusion_matrix = np.array(np.zeros((num_labels,num_labels)),dtype='int16')
    for k in range(len(inputs)):
        # cur_cost[k] = cost(inputs[k], y_pred[k])
        cur_cost[k],output_arr[i][k] = cost(inputs[k], y_pred[k])
        
        pred_index = output_arr[i][k].argmax()
        correct_index = y_pred[k].argmax()
        
        
        if(pred_index == correct_index):
            correct += 1
            correct_labels[correct_index] += 1
    
        confusion_matrix[correct_index][pred_index] += 1
    confusion_total[i] = confusion_matrix
        
    cost_arr.append(cur_cost.mean())
    
    if i % 100 == 0:
        print('%s. %s/%s Correct | Cost: %s | Max: %s @ %d | Min: %s @ %d | %s' % (i,correct,len(inputs),cost_arr[-1], cur_cost.max(), cur_cost.argmax(), cur_cost.min(),cur_cost.argmin(),correct_labels))


end = time.clock()
print("Training time: %f" % (end-start,))




############# PREDICTED







# output_arr[-1] = final predictions




# TESTING

# inputs = np.copy(img_arr)[-1]
# y_pred = filedata['labels'][-1]
# 
# print("Test cost: %s" % (cost(inputs, y_pred)))



# x_axis = np.arange(0, iterations - 1, 10)
# y_axis = cost_arr
# plt.plot(x_axis, y_axis)
# plt.show()

# x = T.matrix('x')
# 
# 
# dot = T.dot(x,w) + b
# 
# 
# tanh = np.tanh(x)
# logistic = 1/(1 + T.exp(-x))
# 
# activation = theano.function([x],tanh)
# log_activation = theano.function([x],logistic)







# 
# p1 = -(output[l2_weight] - correct_output[l2_weight])
# output_err = 
# p2 = (1 - np.power(np.tanh(pred_net[l2_weight]),2))
# p3 = l2_weights[l2_neuron][l2_weight]
# p4 = l2_out[l2_neuron]
# 
# l2_change = p1*p2*p4
# l2_weights_new[l2_neuron][l2_weight] = l2_weights[l2_neuron][l2_weight] + learning_rate*l2_change
# err_temp += p1*p2*p3
# 
# # voila! you get the dEtotal/dl2_out[neuron]
# 
# l1_change = err_temp*(1 - np.power(np.tanh(l2_net[l2_neuron]),2)) * l1_out[l2_neuron]
# l1_weights_new[l1_neuron][l2_neuron] = l1_weights[l1_neuron][l2_neuron] + learning_rate*l1_change
# 
# 
# 








# state = theano.shared(0)
# increment = T.iscalar('increment')
# accumulator = theano.function([increment],state, updates=[(state,state+increment)])
# 




# 
# x = T.vector('x')
# w = T.vector('w')
# b = T.scalar('b')
# 
# 
# z = T.dot(x,w) + b
# a = ifelse(T.lt(z,0),0,1)
# 
# neuron = theano.function([x,w,b],a)
# 
# inputs = [[0,0],[1,1]]
#     
# weights = [1,1]
# 
# bias = -1.5
# 
# for i in range(len(inputs)):
#     t = inputs[i]
#     out = neuron(t,weights,bias)
#     print("The output for x1=%d | x2=%d is %d" % (t[0],t[1],out))























































































