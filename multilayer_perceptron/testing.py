import matplotlib.pyplot as plt
import time
# import matplotlib.rcsetup as rcsetup
import matplotlib.image as mpimg
import numpy as np
import theano
import theano.tensor as T
from theano import function as f
from theano.ifelse import ifelse
import random
np.random.seed(2)
np.set_printoptions(threshold=1000)
np.set_printoptions(precision=6) # default 8
from scipy.interpolate import spline
import winsound
import warnings
warnings.filterwarnings("ignore")
import sys
import timeit
import os

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

class ELM():
    A = []
    R = []
    t = []
    w = []
    X = []
    
    elm_class_one =  np.array([[1,5],[1,6],[1,5.5]])
    elm_class_two =  np.array([[9,2],[8,2],[7.5,2]])
    elm_class_one_T= np.tile(np.array([1,0]),(len(elm_class_one),1))
    elm_class_two_T= np.tile(np.array([0,1]),(len(elm_class_two),1))
    
    elm_A = np.concatenate((elm_class_one,elm_class_two),axis=0)
    elm_A = normalise(elm_A)
    elm_A = np.insert(elm_A,0,1,axis=1)
    elm_T = np.concatenate((elm_class_one_T,elm_class_two_T),axis=0)
    elm_X = np.array([[1,7],[2,8],[2,9],[10,1],[8,2],[9,1]])
    elm_X = normalise(elm_X)
    def __init__(self,A,t,X,lam=0.01):
        self.A = np.insert(normalise(A),0,1,axis=1)
        self.X = normalise(X)
        self.t = t
        self.R =  np.random.uniform(-1,1,(self.A.shape[1],t.shape[1]))
        self.w = np.dot(pinv(np.tanh(np.dot(self.A,self.R)),lam),t)
        self.h = np.dot(self.X,self.w)

def ELM(B,T,X,lam=0.001):
    A = np.insert(B,0,1,axis=1)
    R =  np.random.uniform(-1,1,(A.shape[1],T.shape[1]))
    Ar = np.tanh(np.dot(A,R))
    A_pinv = pinv(Ar,lam)
    w = np.dot(A_pinv,T)
    h = np.dot(X,w)
    return (X,softmax(h),np.argmax(h,axis=1))

def one_hot(index,size):
    lst = [0]*size
    lst[index] = 1
    return lst


def grad_desc(cost, theta):
    
    return theta - (learning_rate*(1) * T.grad(cost, wrt=theta))


num_possible_classes = 5


# seek to minimize ||Xw - t||^2 (i.e. find least-squares solution)
# value of w which does this is A_pinv*T
# A = USV' A_pinv = VS^-1U'
# seek to minimize lambda||w||^2 as well
# i.e. (1/n)*||Xw - t||^2 + lam*||w||^2
# big lambda = small w but more innaccurate




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


class HiddenLayer():
    def __init__(self,input,n_inputs,n_outputs,weights=None,bias=None,activation=T.tanh,rng=np.random.RandomState(2)):
        self.input = input
        if(not weights):
            weights = theano.shared(value=rng.uniform(-1,1,(n_inputs,n_outputs)),name = 'weights')
        if(not bias):
            bias = theano.shared(value=np.zeros((n_outputs,)),name='bias')
            
        self.weights = weights
        self.bias = bias
        
        output = T.dot(input,self.weights) + self.bias
        self.output = output if activation == None else activation(output) 
        self.parameters = [self.weights,self.bias]
 
class OutputLayer():
    def __init__(self,input,n_inputs,n_outputs):
        self.weights = theano.shared(value=np.zeros((n_inputs,n_outputs)),name='weights')
        self.bias = theano.shared(value=np.zeros((n_outputs,)),name='bias')
        self.output = T.nnet.nnet.softmax(T.dot(input,self.weights)+self.bias)
        self.predicted_class = T.argmax(self.output,axis=1)
        self.parameters = [self.weights,self.bias]
        self.input = input
    
    def neg_log_likelihood(self,target):
        return -T.mean(T.log(self.output)[T.arange(target.shape[0]),target])     
    
    def errors(self,y):
        return T.mean(T.neq(self.predicted_class,y))        

class Multilayer_Perceptron():
    def __init__(self,input,shape,n_classes,rng):
        self.hidden_layers = [HiddenLayer(input=input,n_inputs=shape[0],n_outputs=shape[1],activation=None,rng=rng)]
        self.output_layer = OutputLayer(self.hidden_layers[-1].output,shape[-1],n_classes)
        self.L1 = abs(self.hidden_layers[-1].weights).sum() + abs(self.output_layer.weights).sum()
        self.L2 = (self.hidden_layers[-1].weights**2).sum() + (self.output_layer.weights**2).sum()
        self.neg_log_likelihood = self.output_layer.neg_log_likelihood
        
        self.parameters = [a.parameters for a in self.hidden_layers] + self.output_layer.parameters
        self.parameters = self.hidden_layers[0].parameters + self.output_layer.parameters
        self.input = input
        self.errors = self.output_layer.errors


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
    
    suffixes = {}
    
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

batch_size = 30
num_classes=5
learning_rate=0.0001     
L1_rg = 0.00
L2_rg = 0.0001

# 
# def test(learning_rate=0.01, L1_weight=0.0001, L2_weight=0.0001, n_epochs=100, batch_size=20, hidden_shape=(500), num_classes=5):

dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_classes,))])
infile = 'filenames5.txt'
filedata = np.loadtxt(infile,dtype=dt)
filenames = [a.decode('UTF-8') for a in filedata['filename']]
filenames.sort(key=lambda x:x[-7:])



data,targets,suffixes = get_images(w=100,h=100,file_list = filenames)

training_size = len(data)//2
per_class = training_size//num_classes
training_size = per_class*num_classes

training_data = np.zeros((num_classes*per_class,len(data[0])))
training_labels = np.zeros((num_classes*per_class,num_classes))
training_set = (training_data,training_labels)
validation_set = [[],[]]

i = 0
ind = 0
next_ind = suffixes[filenames[i][-7:]]['count']
temp = per_class
while i < len(data):
    if(i < temp):
        training_data[ind] = data[i]
        training_labels[ind] = targets[i]
        i += 1
        ind += 1
    
        
    else:
        validation_set[0].append(data[i:next_ind])
        validation_set[1].append(targets[i:next_ind])
        
        i += suffixes[filenames[i][-7:]]['count'] - per_class
        if(i >= len(data)):
            validation_set[0] = np.concatenate(validation_set[0])
            validation_set[1] = np.concatenate(validation_set[1])
            break
        next_ind += suffixes[filenames[i][-7:]]['count']
        temp = i + per_class

training_batches   =  training_size//batch_size 
validation_batches =  len(validation_set[0])//batch_size
    
# 
batch_index = T.lscalar('batch_index')
x = T.dmatrix('x')
y = T.lvector('y')
rng = np.random.RandomState(2)
classifier = Multilayer_Perceptron(input=x,shape=(10000,100),n_classes=5,rng=rng)
cost = classifier.neg_log_likelihood(y) + L1_rg * classifier.L1 + L2_rg + classifier.L2

gradients = [T.grad(cost,param) for param in classifier.parameters]

updates = [(param, param - learning_rate*gparam) for param, gparam in zip(classifier.parameters,gradients)]

validation_set_x = theano.shared(value=validation_set[0])
validation_set_y = theano.shared(value=validation_set[1].argmax(axis=1))
training_set_x = theano.shared(value=training_set[0])
training_set_y = theano.shared(value=training_set[1].argmax(axis=1))


validate = theano.function(
    inputs = [batch_index],
    outputs= classifier.errors(y),
    givens={
    x: validation_set_x[batch_index * batch_size: (batch_index+1)*batch_size],
    y: validation_set_y[batch_index * batch_size: (batch_index+1)*batch_size]}
    )

train = theano.function(
    inputs = [batch_index],
    outputs = cost,
    updates = updates,
    givens = {
    x: training_set_x[batch_index * batch_size: (batch_index+1)*batch_size],
    y: training_set_y[batch_index * batch_size: (batch_index+1)*batch_size]}
    )

























patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                        # found
improvement_threshold = 0.995  # a relative improvement of this much is
                                # considered significant
validation_frequency = min(training_batches, patience // 2)
                                # go through this many
                                # minibatche before checking the network
                                # on the validation set; in this case we
                                # check every epoch

best_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()
n_epochs = 100000
epoch = 0
done_looping = False
loss_arr = []
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(training_batches):

        minibatch_avg_cost = train(minibatch_index)
        # iteration number
        iter = (epoch - 1) * training_batches + minibatch_index

        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = [validate(i) for i
                                    in range(validation_batches)]
            this_validation_loss = np.mean(validation_losses)
            loss_arr.append(this_validation_loss)

            print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                (
                    epoch,
                    minibatch_index + 1,
                    training_batches,
                    this_validation_loss * 100.
                )
            )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if (
                    this_validation_loss < best_validation_loss *
                    improvement_threshold
                ):
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                # test_losses = [test_model(i) for i
                #                 in range(n_test_batches)]
                # test_score = np.mean(test_losses)

               ##   print(('     epoch %i, minibatch %i/%i, test error of '
                #         'best model %f %%') %
                #         (epoch, minibatch_index + 1, training_batches,
                #         test_score * 100.))

        if patience <= iter:
            done_looping = True
            break

end_time = timeit.default_timer()
print(('Optimization complete. Best validation score of %f %% '
        'obtained at iteration %i') %
        (best_validation_loss * 100., best_iter + 1))

print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)























        



# 

# num = int(num_images/num_classes)
# # selected_imgs = [(images[a][0][:num],[images[a][1] for x in range(num)]) for a in range(num_classes)]
# 
# selected_imgs = np.concatenate(np.array([images[a][0][:num] for a in range(num_classes)]))
# 
# img_arr = get_images(selected_imgs,img_x,img_y)
# img_arr = np.insert(img_arr,0,1,axis=1)
# 
# def print_outputs(main,img_arr):
#     for img in img_arr:
#         main.load_input(img)
#         main.fprop()
#         print("[%7s %7s] | [%7s %7s]" %(("%.4f" % main.layers[-1].softmax()[0]),("%.4f" % main.layers[-1].softmax()[1]),("%.4f" % main.layers[-1].output()[0]),("%.4f" % main.layers[-1].output()[1])  ) )
# 
# 
# def float_str(arr,precision,brace):
#     out_str = brace[0]+" "
#     out_str += ("%%.%df, " % (precision)) * len(arr)
#     out_str = out_str[:-2] % tuple(arr)
#     out_str += " "+brace[1]
#     return out_str
# 
# 
# correct = 0
# 
# 
# loss_results = []
# 
# # main = Multilayer_Perceptron(layers_tuple)
# # in_arr = np.ones((input_size))
# # 
# # 
# # for n in range(num_images):
# # 
# #     img_class = selected_imgs[n][-7:]
# #     target = suffixes[img_class]['label']
# #     
# #     main.load(img_arr[n],target)
# #     results = main.fprop2(1)
# #     
# #     
# #     correct_class = np.argmax(suffixes[img_class]['label'])
# #     
# #     loss_results.append((main.cost(),target))
# #     
# #     if(results.argmax() == correct_class):
# #         correct +=1
# 
#     
#     
# 
#     # print(float_str(softmax(results),2,"[]") + " | " + str(target))
# 
























































