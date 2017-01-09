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

class Node():
    
    activate = np.tanh
    act_str = ""
    in_val = T.scalar('in_val')
    out_val = T.scalar('out_val')
    
    def __init__(self,in_val=0,activation=T.tanh):
        
        self.in_val = in_val
        self.activate = activation
        
        self.out_val = self.activate(in_val)

    def __str__(self):
        return str(self.out_val)
        
class Layer():
    inputs = []
    outputs = []
    nodes = []
    activation = []
    act_str = ""
    
    
    
    def __init__(self,node_inputs,activation='none',bias = 0.0):
        
        self.bias = 0.0
        self.inputs = node_inputs
        self.act_str = activation
        self.activation = activation_functions[activation]
        
        self.outputs = self.activation(self.inputs)
        if(self.act_str == 'soft'):
            self.outputs = self.outputs[0]
        self.nodes = [self.inputs,self.outputs]
        


    @classmethod
    def fromarr(cls,arr,act='tanh'):
        
        # nodes = [Node(in_val=a,activation=activation_functions[act]) for a in arr]
        return cls(arr,act)
        
    @classmethod
    def fromsize(cls,size,act='tanh'):
        # nodes = [Node(activation=activation_functions[act]) for a in range(size)]
        arr = np.array([0 for a in range(size)])
        return cls(arr,act)
    
    @classmethod
    def copy(cls,a):
        
        temp = [Node(in_val=i,activation=a.activation) for i in a.output()]

        return cls(temp,activation=a.act_str)
    
    def in_vals(self):
        return none(self.nodes)
    
    def output(self):
        
        
        return self.outputs
        
    def softmax(self):
        return softmax([a.in_val for a in self.nodes])
        
    def __str__(self):
        return '-'.join(map(str,self.nodes))


        
class Multilayer_Perceptron():
    test_layer = []
    
    
    def __init__(self,shape):
        self.input_size  = shape[0]
        self.output_size = shape[-1]
        
        weights =  [theano.shared(value=np.random.uniform(-1,1,a).transpose()) for a in zip(shape[1:],shape)]
        self.weights = weights
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
  
        
    def print_layer(self,num):
        layer = self.layers[num]
        out_str = ""
        for node in range(self.shape[num]):
            out_str += "%8s | %8s %s\n" % ("%.4f" % (layer.inputs[node]),"%.4f" % layer.outputs[node],layer.act_str)
        print(out_str)

    def load(self,input_arr,target_arr):
        self.layers[0] = Layer.fromarr(input_arr,act='none')
        self.inputs = np.array(input_arr)
        self.target = np.array(target_arr)
        

    def fprop(self,r = False,loss=True):
        new_layers = [self.layers[0]]
        o = T.dvector('o')
        w = T.dmatrix('w')
        act = 'tanh'
        
        l = T.dot(o,w)
        u = theano.function([o,w],l)
        
        temp = u(self.layers[0].outputs,self.weights[0].get_value())
        
        temp_l = Layer(temp,act)
        
        new_layers.append(temp_l)
        
        
        for i in range(1,len(self.layers)-2):
            temp = u(np.array(temp),self.weights[1].get_value())
            temp_l = Layer(temp,act)
            new_layers.append(temp_l)
            

        act = 'sigm'
        temp = u(np.array(temp),self.weights[-1].get_value())
        temp_l = Layer(temp,act)
        new_layers.append(temp_l)
        
        
        self.layers = new_layers

        if(loss):
            self.x = np.array(self.layers[-1].outputs)
            x = self.x
            y = self.target
            self.cross_entropy_loss = T.nnet.nnet.categorical_crossentropy(x,y)
            self.L1 = abs(x).sum()
            self.L2 = (x**2).sum()
            self.cst = self.L1 + self.L2 + self.cross_entropy_loss
            
            

        if(r):
            return main.layers[-1].outputs
    
    def fprop2(self,r = False,loss=True):
        new_layers = [self.layers[0]]
        o = T.dvector('o')
        w = T.dmatrix('w')
        act = 'tanh'
        
        
       
        
        temp = T.dot(self.layers[0].outputs,self.weights[0].get_value())
        
        temp_l = Layer(temp,act)
        
        new_layers.append(temp_l)
        
        
        for i in range(1,len(self.layers)-2):
            temp = T.dot(np.array(temp),self.weights[1].get_value())
            temp_l = Layer(temp,act)
            new_layers.append(temp_l)
            

        act = 'tanh'
        temp = T.dot(np.array(temp),self.weights[-1].get_value())
        temp_l = Layer(temp,act)
        new_layers.append(temp_l)
        
        
        self.layers = new_layers

        if(loss):
            self.x = np.array(self.layers[-1].outputs)
            x = self.x
            y = self.target
            self.cross_entropy_loss = T.nnet.nnet.categorical_crossentropy(x,y)
            self.L1 = abs(x).sum()
            self.L2 = (x**2).sum()
            self.cst = self.L1 + self.L2 + self.cross_entropy_loss
            
            

        if(r):
            return main.layers[-1].outputs
    
    
    def log_regression(self):
        
        
        
        return T.nnet.softmax(T.dot(self.layers[-2].inputs, self.weights[-1]))
    def cost(self):
        return self.cst
        
    def __str__(self):
        h_depth = 1
        temp_str = ""
        for i in self.shape[1:-1]:
            temp_str += "Hidden Layer %s: %d\n" % (h_depth,i)
            h_depth += 1            
        out_str = "Multilayer Perceptron\n%-14s: %d\n%-14s: %d\n%s%-14s: %d" % ("Hidden layers",len(self.shape[1:-1]),"Input size",self.input_size,temp_str,"Output size",self.output_size) 
        # out_str += '\n'.join(map(str,self.layers))
        return out_str



        
num_classes = 5
num_images = 60

output_size = num_classes
img_x = img_y = 20
input_size = 1+img_x*img_y
X = np.zeros((num_images,img_x*img_y))
X = np.insert(X,0,1,axis=1)
hidden_sizes = [10,10]

layers_tuple = [input_size,output_size]
[layers_tuple.insert(-1,a) for a in hidden_sizes]
layers_tuple = tuple(layers_tuple)

x = T.dvector('x')
y = T.dvector('y')

tanh = theano.function([x],T.tanh(x))
sign = theano.function([x],T.sgn(x))
none = theano.function([x],x)
soft = theano.function([x],T.nnet.nnet.softmax(x))
sigm = theano.function([x],T.nnet.nnet.sigmoid(x))

activation_functions = {'tanh':tanh,'sign':sign,'none':none,'soft':soft,'sigm':sigm}


def one_hot(index,size):
    lst = [0]*size
    lst[index] = 1
    return lst
    
    




# dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_labels,))])

dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_classes,))])
infile = 'filenames5.txt'
filedata = np.loadtxt(infile,dtype=dt)
filenames = [a.decode('UTF-8') for a in filedata['filename']]
filenames.sort(key=lambda x:x[-7:])
suffixes = {}

for f in filenames:
    suffixes[f[-7:]] = suffixes.get(f[-7:], 0) + 1

ind = 0
for i in suffixes:    
    suffixes[i] = {"count":suffixes[i],"label":one_hot(ind,num_classes)}
    ind += 1

images=[]

i = 0
for key in suffixes.keys():
    target_temp = num_classes*[0]
    target_temp[i] = 1
    temp = ([a for a in filenames if a[-7:] == key],target_temp)
    images.append(temp)
    i+=1

def get_images(file_list,w,h):
    infile = 'filenames5.txt'
    folder = 'tiffs5/'
    abspath = 'C:/Users/Roan Song/Desktop/thesis/'
    
    img_arr = np.zeros((len(file_list),h*w))
    i = 0
    for fname in file_list:
        img = mpimg.imread(abspath + folder + fname)
        IN_HEIGHT = img.shape[0]
        IN_WIDTH = img.shape[1]
        
        img = pad_img(img,h,w,IN_HEIGHT,IN_WIDTH)
        oneD = img.reshape(h * w)
        oneD = normalise(oneD)
        img_arr[i] = oneD   
        i+=1 
    return img_arr
    
num = int(num_images/num_classes)
# selected_imgs = [(images[a][0][:num],[images[a][1] for x in range(num)]) for a in range(num_classes)]

selected_imgs = np.concatenate(np.array([images[a][0][:num] for a in range(num_classes)]))

img_arr = get_images(selected_imgs,img_x,img_y)
img_arr = np.insert(img_arr,0,1,axis=1)

def print_outputs(main,img_arr):
    for img in img_arr:
        main.load_input(img)
        main.fprop()
        print("[%7s %7s] | [%7s %7s]" %(("%.4f" % main.layers[-1].softmax()[0]),("%.4f" % main.layers[-1].softmax()[1]),("%.4f" % main.layers[-1].output()[0]),("%.4f" % main.layers[-1].output()[1])  ) )


def float_str(arr,precision,brace):
    out_str = brace[0]+" "
    out_str += ("%%.%df, " % (precision)) * len(arr)
    out_str = out_str[:-2] % tuple(arr)
    out_str += " "+brace[1]
    return out_str


correct = 0


loss_results = []

main = Multilayer_Perceptron(layers_tuple)
in_arr = np.ones((input_size))


for n in range(num_images):

    img_class = selected_imgs[n][-7:]
    target = suffixes[img_class]['label']
    
    main.load(img_arr[n],target)
    results = main.fprop2(1)
    
    
    correct_class = np.argmax(suffixes[img_class]['label'])
    
    loss_results.append((main.cost(),target))
    
    if(results.argmax() == correct_class):
        correct +=1

    
    

    print(float_str(softmax(results),2,"[]") + " | " + str(target))
print(correct)

# update_weights = theano.function([],updates=[
#                                 (main.weights[-1],main.weights[-1]*0.01 - T.grad(cst,wrt=main.weights[-1])
# ])

# main.load_input(img_arr[0])

# main.fprop()
























































