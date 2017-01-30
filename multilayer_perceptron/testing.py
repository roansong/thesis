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
from collections import OrderedDict
import utilities as u
import knn as knn

t1 = timeit.default_timer()

class HiddenLayer():
    """
    This class represents a hidden layer of neurons
    It takes an array of inputs, applies an activation function to them, and returns the output
    """
    def __init__(self,input,n_inputs,n_outputs,weights=None,bias=None,activation=T.tanh,rng=np.random.RandomState(2)):
        """
        Initialise the hidden layer
        
        input      --- a vector containing the input values
        n_inputs   --- number of neurons feeding into the hidden layer
        n_outputs  --- number of neurons in the next layer
        weights    --- weights applied to the inputs and outputs of the hidden layer (default: None)
        bias       --- bias applied to the output values (default: None)
        activation --- activation function to be applied to neuron inputs (default: tanh)
        rng        --- seeded random number generator (default: np.random.RandomState(2))
        """
        
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
    """
    This class is a logistic regression layer for use at the output of a neural network
    """
    def __init__(self,input,n_inputs,n_outputs):
        """
        Initialise the output layer
        
        input     --- a vector containing the input values
        n_inputs  --- number of neurons feeding into the layer
        n_outputs --- number of classes
        """
        
        self.weights = theano.shared(value=np.zeros((n_inputs,n_outputs)),name='weights')
        self.bias = theano.shared(value=np.zeros((n_outputs,)),name='bias')
        self.output = T.nnet.nnet.softmax(T.dot(input,self.weights)+self.bias)
        self.predicted_class = T.argmax(self.output,axis=1)
        self.parameters = [self.weights,self.bias]
        self.input = input
    
    def neg_log_likelihood(self,target):
        """
        Returns the negative log likelihood between the classifier's output and a target
        
        target --- correct output
        """
        return -T.mean(T.log(self.output)[T.arange(target.shape[0]),target])     
    
    def errors(self,target):
        """
        Returns the average error between the predicted class and the target class
        
        target --- correct output
        """
        return T.mean(T.neq(self.predicted_class,target))        

class Multilayer_Perceptron():
    def __init__(self,input,shape,num_classes,rng):
        """
        A multilayer perceptron class
        
        input       --- a vector containing the input values
        shape       --- a tuple describing the shape of the classifier and its hidden layers
                        each element in the tuple specifies the number of neurons per layer
        num_classes --- the number of classes in the dataset
        rng         --- seeded random number generator
        """
        
        
        self.hidden_layers = []
        self.hidden_layers.append(
    HiddenLayer(input=input,n_inputs=shape[0],n_outputs=shape[1],activation=None,rng=rng))
        for i in range(2,len(shape)):
            self.hidden_layers.append(
        HiddenLayer(input=self.hidden_layers[-1].output,n_inputs=shape[i-1],n_outputs=shape[i],activation=None,rng=rng)

        )
        
        
        
        
        self.output_layer = OutputLayer(self.hidden_layers[-1].output,shape[-1],num_classes)
        self.L1 = abs(self.output_layer.weights).sum()
        self.L2 = (self.output_layer.weights**2).sum()
        self.parameters = self.output_layer.parameters
        for a in self.hidden_layers:
            self.L1 += abs(a.weights).sum()
            self.L2 += (a.weights**2).sum()
            self.parameters += a.parameters
        
        
        
        self.neg_log_likelihood = self.output_layer.neg_log_likelihood
        
        
        
        self.input = input
        self.errors = self.output_layer.errors
        self.predicted_class = self.output_layer.predicted_class
        self.weights = [a.weights for a in self.hidden_layers]
        self.weights.append(self.output_layer.weights)
        self.shape = shape
        self.rng = rng
        
    def test(self,learning_rate=0.01, L1_rg=0.0000, L2_rg=0.1, n_epochs=100, batch_size=30, print_val = True, quick=True):
        """
        Test regime for the perceptron, used to train it and store performance metrics.
        Some parts are modified from a deeplearning.net tutorial
        
        learning_rate --- scaling factor for gradient descent (default: 0.01)
        L1_rg         --- influence of weights' L1 norm on the classifier cost (default: 0.0)
        L2_rg         --- influence of weights' L2 norm on the classifier cost (default: 0.1)
        n_epochs      --- number of iterations over the dataset to be done during training (default: 100)
        batch_size    --- designates the minibatch size for stochastic gradient descent (default: 30)
        print_val     --- if True, prints the current performance of the model every few epochs (default: False)
        quick         --- if True, stops the classifier from training further if test error is zero (default: True)
        """
        
        start_time = timeit.default_timer()
        batch_index = T.lscalar('batch_index')
        rng = self.rng
        
        cost = self.neg_log_likelihood(y) + L1_rg * self.L1 + L2_rg * self.L2
        correct_classes = self.predicted_class
        
        gradients = [T.grad(cost,param) for param in self.parameters]
        updates = [(param, param - learning_rate*gparam) for param, gparam in zip(self.parameters,gradients)]
        
        validation_set_x = theano.shared(value=validation_set[0])
        validation_set_y = theano.shared(value=validation_set[1].argmax(axis=1))
        training_set_x = theano.shared(value=training_set[0])
        training_set_y = theano.shared(value=training_set[1].argmax(axis=1))
        test_set_x = theano.shared(value=test_set[0])
        test_set_y = theano.shared(value=test_set[1].argmax(axis=1))
        
        ind_test_set_x = theano.shared(value=test_set[0][0:1])
        ind_test_set_y = theano.shared(value=test_set[1][0:1].argmax(axis=1))
        
        validate = theano.function(
            inputs = [batch_index],
            outputs= self.errors(y),
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
        
        test_model = theano.function(
            inputs = [batch_index],
            outputs = self.errors(y),
            givens = {
            x: test_set_x[batch_index * batch_size: (batch_index+1)*batch_size],
            y: test_set_y[batch_index * batch_size: (batch_index+1)*batch_size]}
            )
            
        test = theano.function(
        inputs = [],
        outputs = [self.predicted_class,y],
        givens = {
            x: test_set_x,
            y:test_set_y
        }
        ) 
        
        individual_test = theano.function(
        inputs = [],
        outputs = [self.predicted_class,y],
        givens = {
            x: ind_test_set_x,
            y: ind_test_set_y
        }
        ) 
        
        evaluate_train = theano.function(
        inputs = [],
        outputs = [self.predicted_class,y],
        givens = {
            x: training_set_x,
            y:training_set_y
        }
        )
        
        evaluate_validation = theano.function(
        inputs = [],
        outputs = [self.predicted_class,y],
        givens = {
            x: validation_set_x,
            y:validation_set_y
        }
        )
 
        patience = 1000
        patience_increase = 2
        improvement_threshold = 0.999
        improvement_threshold2 = 0.99
        validation_frequency = min(training_batches, patience // 2)
 
        best_validation_loss = np.inf
        best_avg_cost = np.inf
        best_test_err = np.inf
        best_iter = 0
        test_score = 100.
        
        n_epochs = 1000
        epoch = 0
        done_looping = False
        loss_arr = []
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(training_batches):
        
                minibatch_avg_cost = train(minibatch_index)
                
                iter = (epoch - 1) * training_batches + minibatch_index
        
                if ((iter + 1) % validation_frequency == 0):
                    # compute zero-one loss on validation set
                    validation_losses = [validate(i) for i
                                            in range(validation_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    loss_arr.append(this_validation_loss)
                    
                    if print_val and ((iter + 1) % (validation_frequency*100) == 0):
                        
                        print(
                            'epoch %i,  validation err %f %%, test err %f %%' %
                            (
                                epoch,
                                this_validation_loss * 100.,
                                test_score*100.
                            )
                        )

                    if minibatch_avg_cost < best_avg_cost:
                        if (
                            minibatch_avg_cost < best_avg_cost *
                            improvement_threshold2
                        ):
                            patience = max(patience, iter * patience_increase)
        
                        best_avg_cost = minibatch_avg_cost

                    if this_validation_loss < best_validation_loss:
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)
        
                        best_validation_loss = this_validation_loss
                        
                    test_losses = [test_model(i) for i
                                    in range(test_batches)]
                    test_score = np.mean(test_losses)
    
                    if(test_score < best_test_err):
                        patience = max(patience, iter * patience_increase)
                
                        self.best_weights = [l.get_value() for l in self.weights]
                        best_iter = iter
                        best_test_err = test_score
                
                if patience <= iter:
                    done_looping = True
                    break    
            if(quick and np.isclose(test_score, 0.0)):
                        break    
                
        
        end_time = timeit.default_timer()
        
        
        
        
        a,b = evaluate_train()
        c,d = evaluate_validation()
        e,f = test()
        
        self.results = (best_validation_loss*100, best_iter +1, (end_time-start_time)/60,learning_rate)
        self.train_confusion = u.confusion_matrix(a,b,num_classes)
        self.validation_confusion = u.confusion_matrix(c,d,num_classes)
        self.test_confusion = u.confusion_matrix(e,f,num_classes)
        
        
        
        
        # print("Complete.")
        # print("Training cost: %.3f"%
        # (minibatch_avg_cost),file=sys.stderr)
        # print("Training error: %.3f"%
        # (100*(1 - (np.diag(self.train_confusion).sum()/len(training_set[0])))),file=sys.stderr)
        # print("Validation error: %.3f"%(best_validation_loss*100),file=sys.stderr)
        # print("Test error: %.3f"%(best_test_err*100),file=sys.stderr)
        # print("Best iteration: %d"%(best_iter +1),file=sys.stderr)
        # 
        # 
        # print("Confusion matrices:")
        # print("Training")
        # print(self.train_confusion)
        # print("Validation")
        # print(self.validation_confusion)
        # print("Testing")
        # print(self.test_confusion)
        # print("="*60)
        
        
        
        ui = timeit.default_timer()
        individual_test()
        io = timeit.default_timer()
        
        # print("Individual classification: %f" %
        # ((io-ui)/2))
        
        
        
        # print('The code for file ' + os.path.split(__file__)[1] +' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)
        
        metrics = [0]*7
        metrics[0] = classifier.shape[1:]
        metrics[1] = (io-ui)/2
        metrics[2] = best_test_err*100
        metrics[3] = best_validation_loss*100
        metrics[4] = end_time - start_time
        metrics[5] = 100 - (np.diag(self.train_confusion).sum()/num_classes)
        metrics[6] = minibatch_avg_cost
        
        self.metrics = metrics
        
        return

IMG_WIDTH = 194
IMG_HEIGHT = 194

batch_size = 30
num_classes=8
learning_rate=0.01     




dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_classes,))])
infile = str(num_classes)
filedata = np.loadtxt(infile+'.txt',dtype=dt)
filenames = [a.decode('UTF-8') for a in filedata['filename']]
filenames.sort(key=lambda x:x[-7:])



data,targets,suffixes = u.get_images(w=IMG_WIDTH,h=IMG_HEIGHT,file_list = filenames,threshold = True,noise=True,num_classes = num_classes)


# training_set, validation_set, test_set,indices = u.gen_sets(data,targets,85,5,10)
training_set, validation_set, test_set,indices = u.gen_sets(data,targets,85,5,10)
training_batches   =  len(training_set[0])//batch_size 
validation_batches =  len(validation_set[0])//batch_size
test_batches       =  len(test_set[0])//batch_size


x = T.dmatrix('x')
y = T.lvector('y')

# k,results = testKNN(training_set,validation_set,test_set)
# np.save("full_classk_conf",k_conf)
# print(k_conf)
# Single layer best learning rate: 0.01

L2 = 0.1
L1 = 0
# size = [(IMG_WIDTH*IMG_HEIGHT,10),(IMG_WIDTH*IMG_HEIGHT,10,10),(IMG_WIDTH*IMG_HEIGHT,50),(IMG_WIDTH*IMG_HEIGHT,50,50)]
# size = [(IMG_WIDTH*IMG_HEIGHT,5),
# (IMG_WIDTH*IMG_HEIGHT,10),
# (IMG_WIDTH*IMG_HEIGHT,20),
# (IMG_WIDTH*IMG_HEIGHT,5,1),
# (IMG_WIDTH*IMG_HEIGHT,5,5),
# (IMG_WIDTH*IMG_HEIGHT,10,5),
# (IMG_WIDTH*IMG_HEIGHT,10,10),
# (IMG_WIDTH*IMG_HEIGHT,5,5,5),
# (IMG_WIDTH*IMG_HEIGHT,10,10,10)
# ]

size = [(IMG_WIDTH*IMG_HEIGHT,5,5)]
# 
for s in size:
    # learning_rate = 1/(10**len(s))
    classifier = Multilayer_Perceptron(x,
                            s,
                            num_classes=num_classes,
                            rng=np.random.RandomState(1)
                            )
    # learning_rate = 1/(10*sum(classifier.shape[1:]))
    
    
    classifier.test(
        learning_rate=learning_rate,
        L2_rg=L2,
        L1_rg= L1,
        n_epochs=2000,
        print_val=False,
        quick=True)
    print(
    "Learning rate       : %.5f\n"%(learning_rate)+(
    "Hidden Layer Sizes  : %s\n" + 
    "Classification Time : %8f\n" +
    "Classification Error: %.3f\n" +
    "Validation Error    : %.3f\n" +
    "Training Time       : %.4f\n" +
    "Training Error      : %.3f\n" +
    "Training Cost       : %.4f")%
    (tuple(classifier.metrics)))
    print("="*40)
    print(classifier.train_confusion)
    print(classifier.validation_confusion)
    print(classifier.test_confusion)
# 
#   
# start = timeit.default_timer()
# kdata = np.concatenate((training_set[0],validation_set[0]))
# ktargets = np.concatenate((training_set[1],validation_set[1]))
# k_indices = np.concatenate((indices[0],indices[1]))
# k3 = knn.KNN(kdata,ktargets)
# start = timeit.default_timer()
# k3.run(test_set[0][0],k=1,y=test_set[1][0])
# end = timeit.default_timer()
# print(end - start)



# start = timeit.default_timer()
# k,r = knn.testKNN(training_set,validation_set,test_set,indices,k=1,filename="100x100.npy")
# mid = timeit.default_timer()
# # k2,r2 = knn.testKNN(training_set,validation_set,test_set,indices,k=3,filename=None)
# end = timeit.default_timer()
# nn_time = mid-start
# # knn_time = end-mid
# print(nn_time)
# print(knn_time)
# 
# a = list(zip(*r))[0]
# b = list(zip(*r))[2]
# conf1 = u.confusion_matrix(a,b,num_classes)
# 
# a = list(zip(*r2))[0]
# b = list(zip(*r2))[2]
# conf2 = u.confusion_matrix(a,b,num_classes)
#  Time an individual run for nn and knn
# 
#  Run two MLPs on the same data here










# print("%.4fm calculating K values" % ((end_time-start_time)/60.))
# 
# 
# k_ascending = sorted(results,key=lambda x:x[1])[-1]
# 
# print("Top 3 K values: %d, %d, %d" %(k_ascending[0],k_ascending[1],k_ascending[2]))
# 


# 
# k_arr = np.arange(1,240,2)
# k_err = np.ones((len(k3.k_values)))*100 - list(zip(*k3.k_values))[2]
# plt.figure(1)
# plt.subplot(2,1,1)
# x = k_arr
# y = k_err
# plt.axis([min(x), max(x)+1, 0, 100])
# plt.xlabel("K")
# plt.ylabel("Error")
# plt.title("Values of K up to 239")
# plt.plot(x,y)
# plt.subplot(2,1,2)
# x = k_arr[:25]
# y = k_err[:25]
# plt.plot(x,y)
# plt.axis([min(x), max(x), 0, 100])
# plt.ylabel("Error")
# plt.xlabel("K")
# plt.xticks(np.arange(min(x), max(x)+1, 4))
# plt.title("First 25 values of K")
# plt.show()









t2 = timeit.default_timer()
print("Total execution time: %.2f" % (t2-t1))
























