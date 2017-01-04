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
    



batch_size = 50
learning_rate = 0.01

batch_sizes = [50, 100, 150, 200, 500]
learning_rates = [0.01, 0.1, 0.15, 0.2, 0.3, 0.4]
learning_rates = [0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]
batch_sizes = [1, 2, 5, 10, 20, 30, 40, 50]
# 
# batch_sizes = [100]
# learning_rates = [0.1]

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
                       
                       
                       
feed_forward = theano.function(inputs = [x,y], outputs=[cst,output])
feed_forward_update = theano.function(inputs = [x,y], outputs=[cst], 
                                      updates = [
                                      (l1_weights, grad_desc(cst,l1_weights)),
                                      (l2_weights, grad_desc(cst, l2_weights)),
                                      (out_weights, grad_desc(cst, out_weights)) 
                                      ])

l1_delta = theano.function(inputs=[x,y],outputs=grad_desc(cst,l1_weights))
l2_delta = theano.function(inputs=[x,y],outputs=grad_desc(cst,l2_weights))
out_delta = theano.function(inputs=[x,y],outputs=grad_desc(cst,out_weights))                



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




p = T.dscalar('p')
q = T.dscalar('q')

mn = p/q

my_mean = theano.function(inputs=[p,q],outputs=[mn])



# TRAINING

inputs = np.copy(img_arr)
y_pred = filedata['labels']



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





# def run(iterations, print_freq,logfile):
#     # start = time.clock()
#     for i in range(iterations):
#         if i % print_freq == 0:
#             correct = 0
#             correct_labels = np.zeros(num_labels)
#             confusion_matrix = np.array(np.zeros((num_labels,num_labels)),dtype='int16')
#             for k in range(len(inputs)):
#                 cur_cost[k],output_arr[i][k] = feed_forward(inputs[k], y_pred[k])
#                 
#                 # cur_cost[k],output_arr[i][k] = cost(inputs[k], y_pred[k]
#                 pred_index = output_arr[i][k].argmax()
#                 correct_index = y_pred[k].argmax()
#                 
#                 
#                 if(pred_index == correct_index):
#                     correct += 1
#                     correct_labels[correct_index] += 1
#             
#                 confusion_matrix[correct_index][pred_index] += 1
#             
#             
#             confusion_total.append(confusion_matrix)
#                 
#             cost_arr.append((cur_cost.mean(),cur_cost.max(),cur_cost.min()))
#             log_str = '%s. %s/%s Correct | Cost: %s | Max: %s @ %d | Min: %s @ %d | %s' % (i,correct,len(inputs),cost_arr[-1][0], cur_cost.max(), cur_cost.argmax(), cur_cost.min(),cur_cost.argmin(),correct_labels)
#             if i % (print_freq*10) == 0:
#                 print(log_str)
#             logfile.write(log_str+'\n')
#     
#     # end = time.clock()
#         
#     return cost_arr[-1] 


def run_straight(iterations):
    for i in range(iterations):
        for k in range(len(inputs)):
            feed_forward_update(inputs[k],y_pred[k])

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


# run_fname = "logs/%sx%sC%sN%sL1%sL2%sB%sLR%sI%s" % (IMG_WIDTH,IMG_HEIGHT,num_labels,num_images,layer1_size,layer2_size,batch_size,learning_rate,iterations)
# f = open(run_fname + ".log",'w')


# print("Learning rate: %s" % (learning_rate))
# f.write("Learning rate: %s\n" % (learning_rate))
# print("Batch size: %s" % (batch_size))
# f.write("Batch size: %s\n" % (batch_size))
# print("Iterations: %s" % (iterations))
# f.write("Iterations %s\n" % (iterations))


# load_str = ""
# 
# # fin_cost, fin_max, fin_min = run(iterations,print_freq,f)
# # cost_tuples.append((fin_cost,fin_max,fin_min))
# # f.close()
# 
# print("=============================")
# print("Initial Run of 1000 Iterations 0.1 Learning rate")
# 
# learning_rate = 0.1
# results_array = []
# 
# for j in range(10):
#     print(load_str)
#     load_str += "="
#     run_straight(100)
#     correct_guesses, cost_tuple, confusion = test()
#     results_array.append((correct_guesses, cost_tuple, confusion))
#     r_print(correct_guesses, cost_tuple, confusion)
#     weights_arr_straight = export_weights()
# np.save("weights_arr_straight",weights_arr_straight)    
# # load_str = ""
# # print("=============================")
# # print("Second Run of 1000 Iterations 0.01 Learning rate")
# # 
# # learning_rate = 0.01
# # 
# # 
# # for j in range(50):
# #     print(load_str)
# #     load_str += "="
# #     run_straight(20)
# #     correct_guesses, cost_tuple, confusion = test()
# #     results_array.append((correct_guesses, cost_tuple, confusion))
# #     r_print(correct_guesses, cost_tuple, confusion)
# #     weights_arr_straight = export_weights()
#     
# load_str = ""
# load_len = "="*10    
# 
# print("Batch Run of 1000 Iterations 0.001 Learning rate")
# 
# learning_rate = 0.001
# 
# for j in range(10):
#     print(load_len)
#     print(load_str)
#     
#     load_str += "="
#     run_batches(100,100)
#     correct_guesses, cost_tuple, confusion = test()
#     results_array.append((correct_guesses, cost_tuple, confusion))
#     r_print(correct_guesses, cost_tuple, confusion)
#     weights_arr_batches1 = export_weights()
# 
# np.save("weights_arr_batches1",weights_arr_batches1)   
# 
# load_str = ""
# # print("=============================")
# print("Second Batch Run of 1000 Iterations 0.001 Learning rate")
# 
# for j in range(10):
#     print(load_len)
#     print(load_str)
#     load_str += "="
#     run_batches(100,200)
#     correct_guesses, cost_tuple, confusion = test()
#     results_array.append((correct_guesses, cost_tuple, confusion))
#     r_print(correct_guesses, cost_tuple, confusion)
#     weights_arr_batches2 = export_weights()
# np.save("weights_arr_batches2",weights_arr_batches2)  


# np.save(run_fname,weights_arr)
# 
# print("Cost: %s | Max: %s | Min%s" % (fin_cost,fin_max,fin_min))
# if(fin_cost < best_cost):
#     print("Current best: %s " % (fin_cost))
# print()








# fig = plt.figure()
# fig.add_subplot(211)
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,1,2)
# for temp in zip(*cost_tuples):
#      ax2.plot(x_tuples,np.array(temp))
# 
# for temp in zip(*cost_arr):
#      ax1.plot(x_raw,np.array(temp))
# 
# ax1.set_title('Cost after every 100th batch')
# ax1.set_xlabel('Batch number (x100)')
# ax1.set_ylabel('Cost')
# ax2.set_ylabel('Cost')
# ax2.set_xlabel('Epoch (10,000 iterations)')
# ax2.set_title('Cost per epoch')
# ax2.legend(['Average','Max','Min'],loc='upper right')
# ax1.legend(['Average','Max','Min'],loc='upper right')
# plt.show(fig)
# extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('test.pdf',bbox_inches=extent)
# fig.savefig('test.pdf',bbox_inches=extent.expanded(1.1,1.2))







































































