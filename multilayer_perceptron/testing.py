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
    if(len(h.shape) == 2):
        return np.array([np.exp(x)/np.sum(np.exp(x), axis=0) for x in h])
    elif(len(h.shape) == 1):
        return np.exp(h)/np.sum(np.exp(h),axis=0)

def ELM(B,T,X,lam=0.001):
    A = np.insert(B,0,1,axis=1)
    R =  np.random.uniform(-1,1,(A.shape[1],T.shape[1]))
    Ar = np.tanh(np.dot(A,R))
    A_pinv = pinv(Ar,lam)
    w = np.dot(A_pinv,T)
    h = np.dot(X,w)
    return (X,softmax(h),np.argmax(h,axis=1))

elm_class_one =  np.array([[1,5],[1,6],[1,5.5]])
elm_class_two =  np.array([[9,2],[8,2],[7.5,2]])
elm_class_one_T= np.tile(np.array([1,0]),(len(elm_class_one),1))
elm_class_two_T= np.tile(np.array([0,1]),(len(elm_class_two),1))

elm_A = np.concatenate((elm_class_one,elm_class_two),axis=0)
elm_A = normalise(elm_A)
elm_A = np.insert(elm_A,0,1,axis=1)
elm_T = np.concatenate((elm_class_one_T,elm_class_two_T),axis=0)

elm_X = np.array([[1,7],
              [2,8],
              [2,9],
              [10,1],
              [8,2],
              [9,1]
              ])
elm_X = normalise(elm_X)



num_classes = 2


# seek to minimize ||Xw - t||^2 (i.e. find least-squares solution)
# value of w which does this is A_pinv*T
# A = USV' A_pinv = VS^-1U'
# seek to minimize lambda||w||^2 as well
# i.e. (1/n)*||Xw - t||^2 + lam*||w||^2
# big lambda = small w but more innaccurate

num_images = 2

output_size = num_classes
img_x = img_y = 20
input_size = 1+img_x*img_y
X = np.zeros((num_images,img_x*img_y))
X = np.insert(X,0,1,axis=1)
hidden_sizes = [10,10]

layers_tuple = [input_size,output_size]
[layers_tuple.insert(-1,a) for a in hidden_sizes]
layers_tuple = tuple(layers_tuple)

activation_functions = {'tanh':np.tanh,'sign':np.sign,'none':lambda x:x}

class Node():
    
    activate = np.tanh
    act_str = ""
    in_val = 0
    out_val = 0
    
    def __init__(self,in_val=0,activation=np.tanh):
        
        self.in_val = in_val
        self.activate = activation
        
        self.out_val = self.activate(in_val)

    def __str__(self):
        return str(self.out_val)
        
class Layer():
    nodes = []
    activation = np.tanh
    act_str = ""
    
    def __init__(self,nodes,activation='none'):
        self.nodes = nodes
        self.act_str = activation
        self.activation = activation_functions[activation]

    @classmethod
    def fromarr(cls,arr,act='tanh'):
        nodes = [Node(in_val=a,activation=activation_functions[act]) for a in arr]
        return cls(nodes,act)
        
    @classmethod
    def fromsize(cls,size,act='tanh'):
        nodes = [Node(activation=activation_functions[act]) for a in range(size)]
        return cls(nodes,act)
    
    @classmethod
    def copy(cls,a):
        
        temp = [Node(in_val=i,activation=a.activation) for i in a.outputs()]

        return cls(temp,activation=a.act_str)
    
    def outputs(self):
        return [a.out_val for a in self.nodes]
        
    def __str__(self):
        return '-'.join(map(str,self.nodes))
        
class Multilayer_Perceptron():
    input_size  = 0
    output_size = 0
    weights = []
    shape = ()
    layers = []
    
    def __init__(self,shape):
        self.input_size  = shape[0]
        self.output_size = shape[-1]
        self.weights = [np.random.uniform(-1,1,a).transpose() for a in zip(shape[1:],shape)]
        self.shape = shape
        self.init_layers()
        
    def init_layers(self):
        input_l = True
        self.layers = []
        for i in self.shape:
            if(input_l):
                self.layers.append(Layer.fromsize(i,act='none')) 
                input_l = False
            else:
                self.layers.append(Layer.fromsize(i)) 
        self.layers[-1] = Layer.fromsize(self.shape[-1],act='sign')    
        
    def print_layer(self,layer):
        out_str = ""
        for node in range(self.shape[layer]):
            out_str += "%.4f | %.4f %s\n" % (self.layers[layer].nodes[node].in_val,self.layers[layer].nodes[node].out_val,self.layers[layer].act_str)
        print(out_str)

    def load_input(self,input_arr):
        self.layers[0] = Layer.fromarr(input_arr,act='none')
        
    def fprop(self):
        print("Forward propagating")
        
        
        new_layers = [self.layers[0]]
       
        for i in range(len(self.layers)-2):
            
            temp = Layer.fromarr(np.dot(np.array(new_layers[i].outputs()),self.weights[i]) ,act='tanh')
            new_layers.append(temp)
        
        new_layers.append(Layer.fromarr(np.dot(np.array(new_layers[-2].outputs()),self.weights[-1]),act='sign'))
        self.layers = [Layer.copy(u) for u in new_layers]
        print("Complete")
        
    def __str__(self):
        h_depth = 1
        temp_str = ""
        for i in self.shape[1:-1]:
            temp_str += "Hidden Layer %s: %d\n" % (h_depth,i)
            h_depth += 1            
        out_str = "Multilayer Perceptron\n%-14s: %d\n%-14s: %d\n%s%-14s: %d" % ("Hidden layers",len(self.shape[1:-1]),"Input size",self.input_size,temp_str,"Output size",self.output_size) 
        # out_str += '\n'.join(map(str,self.layers))
        return out_str

main = Multilayer_Perceptron(layers_tuple)
in_arr = np.ones((input_size))
main.load_input(in_arr)
main.fprop()









































































