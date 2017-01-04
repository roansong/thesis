import matplotlib.pyplot as plt
import time
# import matplotlib.rcsetup as rcsetup
import matplotlib.image as mpimg
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import random
np.random.seed(2)
np.set_printoptions(threshold=1000)
np.set_printoptions(precision=6) # default 8
from scipy.interpolate import spline
import winsound

IMG_HEIGHT = 100
IMG_WIDTH = 100
num_labels = 2
num_images = 469

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



def layer(X,w,bias=1):
    # b = np.array([bias], dtype=theano.config.floatX)
    # new_X = T.concatenate([X,b])
    # m = T.dot(w, new_X)
    m = T.dot(w.T, X)
    m += bias
    output = T.nnet.sigmoid(m)
    # output = np.tanh(m)
    return output


def grad_desc(cost, theta):
    
    return theta - (learning_rate * T.grad(cost, wrt=theta))



c = T.dvector('c')
x = T.dvector('x')
y = T.dvector('y')
w = T.dmatrix('w')
output = T.dvector('output')



x_arr = T.dmatrix('x_arr')
y_arr = T.dmatrix('y_arr')

layer1_size = 30
layer2_size = 20

num_pixels = IMG_HEIGHT*IMG_WIDTH


l1_weights  = theano.shared(np.array(np.random.rand(num_pixels,layer1_size) , dtype=theano.config.floatX))
l2_weights  = theano.shared(np.array(np.random.rand(layer1_size,layer2_size), dtype=theano.config.floatX))
out_weights = theano.shared(np.array(np.random.rand(layer2_size,num_labels) , dtype=theano.config.floatX))
pred        = theano.shared(np.array(np.zeros(num_labels), dtype=theano.config.floatX))


show_weights = theano.function([],[l1_weights, l2_weights, out_weights])

def export_weights():
    
    
    
    l1, l2, out = show_weights()
    
    l1  = np.array(l1)[0]
    l2  = np.array(l2)[0]
    out = np.array(out)[0]
    
    
    
    return l1, l2, out
    


learning_rate = 0.01



hidden_layer1 = layer(x, l1_weights)
hidden_layer2 = layer(hidden_layer1, l2_weights)
output =        layer(hidden_layer2, out_weights)
cst =           ((output - y)**2).sum()


             
                       
feed_forward = theano.function(inputs = [x,y], outputs=[cst,output])
feed_forward_update = theano.function(inputs = [x,y], outputs=[cst,output], 
                                      updates = [
                                      (l1_weights, grad_desc(cst,l1_weights)),
                                      (l2_weights, grad_desc(cst, l2_weights)),
                                      (out_weights, grad_desc(cst, out_weights)) 
                                      ])

l1_delta = theano.function(inputs=[x,y],outputs=grad_desc(cst,l1_weights))
l2_delta = theano.function(inputs=[x,y],outputs=grad_desc(cst,l2_weights))
out_delta = theano.function(inputs=[x,y],outputs=grad_desc(cst,out_weights))                

#

# update_weights = theano.function(inputs = [x,y], 
#                        outputs=[cst,output], 
#                        updates=[
#                        (l1_weights, grad_desc(cst, l1_weights)),
#                        (l2_weights, grad_desc(cst, l2_weights)),
#                        (out_weights, grad_desc(cst, out_weights))])


l1d = T.dmatrix('l1d')
l2d = T.dmatrix('l2d')
outd = T.dmatrix('outd')

update_weights = theano.function(inputs = [l1d,l2d,outd], 
                       outputs=[], 
                       updates=[
                       (l1_weights, l1d ),
                       (l2_weights, l2d),
                       (out_weights, outd)])




# TRAINING

inputs = np.copy(img_arr)
y_pred = filedata['labels'][:len(inputs)]



iterations = 5000
print_freq = 100

cur_cost = 0
cost_arr = []

cur_cost = np.zeros(len(inputs),dtype=theano.config.floatX)

output_arr = np.zeros((iterations,num_images,num_labels))
# confusion_total = np.array(np.zeros((iterations,num_labels,num_labels)),dtype='int32')
confusion_total = []
confusion_matrix = np.zeros((num_labels,num_labels))



def init_weights():
    l1_weights  = theano.shared(np.array(np.random.rand(num_pixels,layer1_size) , dtype=theano.config.floatX))
    l2_weights  = theano.shared(np.array(np.random.rand(layer1_size,layer2_size), dtype=theano.config.floatX))
    out_weights = theano.shared(np.array(np.random.rand(layer2_size,num_labels) , dtype=theano.config.floatX))
    return


def run_straight(iterations):
    for i in range(iterations):
        for k in range(len(inputs)):
            feed_forward_update(inputs[k],y_pred[k])
    winsound.Beep(300,2000)

def run_batches(iterations,batch_size):
    for i in range(iterations):
        selection = np.random.choice(num_images,batch_size,replace=False)
        l1 = 0
        l2 = 0
        out = 0
        
        for h in range(batch_size):
        
            l1 += l1_delta(inputs[selection][h], y_pred[selection][0])
            
            l2 += l2_delta(inputs[selection][h], y_pred[selection][0])
            
            out += out_delta(inputs[selection][h], y_pred[selection][0])
        
        l1 /= batch_size
        l2 /= batch_size
        out /= batch_size
        
        update_weights(l1,l2,out)
    winsound.Beep(300,2000)
 
def r_print(correct_guesses, cost_tuple, confusion):
    print("=============================")
    print("Correct: %s" % (correct_guesses))
    print("Avg Cost: %s" % (cost_tuple[0]))
    print("Max Cost: %s" % (cost_tuple[1]))
    print("Min Cost: %s" % (cost_tuple[2]))
    print("Confusion Matrix:")
    print(confusion) 
        
    
def test():
    correct = 0
    correct_labels = np.zeros(num_labels)
    confusion_matrix = np.array(np.zeros((num_labels,num_labels)),dtype='int16')
    total_cost = 0
    max = -1.0
    min = 1000.0
    for k in range(len(inputs)):
        temp_cost,temp_output = feed_forward(inputs[k], y_pred[k])
        total_cost += temp_cost
        pred_index = temp_output.argmax()
        correct_index = y_pred[k].argmax()
        
        if(temp_cost < min):
            min = temp_cost
        if(temp_cost > max):
            max = temp_cost
        
        
        if(pred_index == correct_index):
            correct += 1
            correct_labels[correct_index] += 1
    
        confusion_matrix[correct_index][pred_index] += 1
    
    
    confusion_total.append(confusion_matrix)
    avg = total_cost/len(inputs)
    
    
    
    return correct, (avg,max,min), confusion_matrix
    





def run(run_type, iterations = 100, batch_size = 50, lr = 0.01):
    
    learning_rate = lr
    if(run_type == 1):
        run_straight(iterations)
        
    elif(run_type == 2):
        run_batches(iterations,batch_size)
        
    else:
        return

    a,b,c = test()
    r_print(a,b,c)
    return a,b,c

def import_weights(fname):
    if not (fname[-4:] == '.npy'):
        fname += '.npy'
    
    w = np.load(fname)
    l1_weights = theano.shared(w[0])
    l2_weights = theano.shared(w[1])
    out_weights = theano.shared(w[2])








best_cost = 10000

cost_tuples = []
results_array = []
