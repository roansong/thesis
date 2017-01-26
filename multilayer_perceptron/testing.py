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
    def __init__(self,input,shape,num_classes,rng):
        self.hidden_layers = [HiddenLayer(input=input,n_inputs=shape[0],n_outputs=shape[1],activation=None,rng=rng)]
        self.output_layer = OutputLayer(self.hidden_layers[-1].output,shape[-1],num_classes)
        self.L1 = abs(self.hidden_layers[-1].weights).sum() + abs(self.output_layer.weights).sum()
        self.L2 = (self.hidden_layers[-1].weights**2).sum() + (self.output_layer.weights**2).sum()
        self.neg_log_likelihood = self.output_layer.neg_log_likelihood
        
        self.parameters = [a.parameters for a in self.hidden_layers] + self.output_layer.parameters
        self.parameters = self.hidden_layers[0].parameters + self.output_layer.parameters
        self.input = input
        self.errors = self.output_layer.errors
        self.predicted_class = self.output_layer.predicted_class
        self.weights = [self.hidden_layers[-1].weights]
        self.weights.append(self.output_layer.weights)
        self.shape = shape
        self.rng = rng
        
    def test(self,learning_rate=0.001, L1_rg=0.0000, L2_rg=0.0001, n_epochs=100, batch_size=30, print_val = True):
        print("="*60)
        print("Multilayer Perceptron.",
              "Image size: %dpx" %(self.shape[0]),
              "Hidden layer size: %d"%(self.shape[1]),
              "Learning rate: %.5f"%(learning_rate),
              "L1 reg: %.5f"%(L1_rg),
              "L2 reg: %.5f"%(L2_rg),
              "Batch size: %d"%(batch_size),
              sep='\n')
    
    
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
        best_avg_cost = np.inf
        best_test_err = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()
        n_epochs = 1000
        epoch = 0
        done_looping = False
        loss_arr = []
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(training_batches):
        
                minibatch_avg_cost = train(minibatch_index)
                
                # iteration number
                iter = (epoch - 1) * training_batches + minibatch_index
        
                if print_val and ((iter + 1) % validation_frequency == 0):
                    # compute zero-one loss on validation set
                    validation_losses = [validate(i) for i
                                            in range(validation_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    loss_arr.append(this_validation_loss)
                    
                    # if print_val and ((iter + 1) % (validation_frequency*10) == 0):
                        
                        # print(
                        #     'epoch %i, minibatch %i/%i, validation error %f %%' %
                        #     (
                        #         epoch,
                        #         minibatch_index + 1,
                        #         training_batches,
                        #         this_validation_loss * 100.
                        #     )
                        # )
                        # print(minibatch_avg_cost)
                    # if we got the best validation score until now
                    
                    # if this_validation_loss < best_validation_loss:
                    if minibatch_avg_cost < best_avg_cost:
                        #improve patience if loss improvement is good enough
                        if (
                            minibatch_avg_cost < best_avg_cost *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)
        
                        best_avg_cost = minibatch_avg_cost
                    
        
                        
                    if this_validation_loss < best_validation_loss:
                        if (
                            minibatch_avg_cost < best_avg_cost *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)
        
                        best_validation_loss = this_validation_loss
                        
                        
                        
                        # test it on the test set
                    test_losses = [test_model(i) for i
                                    in range(test_batches)]
                    test_score = np.mean(test_losses)
    
                    if(test_score < best_test_err):
                        
                
                        # print(('epoch %i, minibatch %i/%i, test error of %f %%') %
                        #         (epoch, minibatch_index + 1, training_batches,
                        #         test_score * 100.))
                        self.best_weights = [l.get_value() for l in self.weights]
                        best_iter = iter
                        best_test_err = test_score
                        
                if patience <= iter:
                    done_looping = True
                    break
        
        end_time = timeit.default_timer()
        
        
        
        
        a,b = evaluate_train()
        c,d = evaluate_validation()
        e,f = test()
        
        self.results = (best_validation_loss*100, best_iter +1, (end_time-start_time)/60,learning_rate)
        self.train_confusion = u.confusion_matrix(a,b,num_classes)
        self.validation_confusion = u.confusion_matrix(c,d,num_classes)
        self.test_confusion = u.confusion_matrix(e,f,num_classes)
        
        
        
        
        print("Complete.")
        print("Validation score: %.3f"%(best_validation_loss*100),file=sys.stderr)
        print("Test score: %.3f"%(best_test_err*100),file=sys.stderr)
        print("Best iteration: %d"%(best_iter +1),file=sys.stderr)
        
        
        print("Confusion matrices:")
        print("Training")
        print(self.train_confusion)
        print("Validation")
        print(self.validation_confusion)
        print("Testing")
        print(self.test_confusion)
        print("="*60)
        
        
        
        
        
        print('The code for file ' + os.path.split(__file__)[1] +' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)
        
        
        
        return

class KNN():
    def __init__(self,input,targets):
        # note that inputs and targets are of type numpy.ndarray
        self.data = input
        self.targets = targets
     
    
    def initD2(self,filename=None,size=None,indices=None):
        if(filename==None):
            D2 = np.zeros((len(self.data),len(self.data)))
            for i in range(len(self.data)):
                for l in range(i,len(self.data)):
                    cost = 0
                    if(i != l):
                        for j in range(len(self.data[i])):
                            cost += pow(self.data[i][j] - self.data[l][j],2)
                    D2[i][l] = D2[l][i] = cost      
            
                progress_bar(i,len(self.data))    
        else:
            D2 = np.load(filename)
            if(indices != None):
                
                temp = np.zeros((len(indices),len(indices)))
                for y in range(len(indices)):
                    for x in range(len(indices)):
                        temp[y,x] = D2[indices[y],indices[x]]

                D2 = temp
                
            elif(size):
                D2 = D2[:size,:size]
            
        self.D2 = D2
        
    def test(self,k_arr):
        results = []
        correct = np.zeros((len(k_arr))) 
             
        for img in range(len(self.data)):
            # the 0th item in the list is the image to itself, so it is removed
            # put K in HERE you knob, instead of recalculating and sorting each time
            costs = sorted(list(zip(self.D2[img],self.targets.argmax(axis=1))))
            
            pred_lst = np.zeros((len(k_arr)))
            confidence = np.zeros((len(k_arr)))
            accuracy = np.zeros((len(k_arr)))
            ind = 0
            for k in k_arr:
                
                
                pred = list(zip(*costs[:k]))[1][1:]
            
                predicted_class = 0
                max = 0 
                for i in pred: 
                    cnt = 0
                    for l in pred:
                        
                        if(i == l):
                            cnt += 1
                    if(cnt > max):
                        max = cnt
                        predicted_class = i
                # pred_lst[ind] = predicted_class
                if(predicted_class == self.targets[img].argmax()):
                    correct[ind] += 1
                confidence[ind] += max/k * 100
                accuracy[ind] = correct[ind]/len(self.data) * 100
                ind +=1 
                
        
        for x in range(ind):
            results.append([k_arr[x],correct[x],accuracy[x],confidence[x]])
        
        self.results = results  
        self.pred = list(zip(*costs[:k]))[1][1:]
        self.costs= list(zip(*costs[:k]))[0][1:]
        
        return np.array(results)
    
    def run(self,x,y,k):
        temp = []
        
        for img in range(len(self.data)):
            if(np.equal(self.data[img],x).all()):
                continue
            cost2 = 0
            for px in range(len(self.data[img])):
                
                cost2 += pow(self.data[img][px] - x[px],2)
            temp.append((cost2,self.targets[img].argmax()))
        
        temp = sorted(temp)[:]
        
        
        cost = list(zip(*temp[:k]))[0]
        pred = list(zip(*temp[:k]))[1]
        
        max = 0
        predicted_class = []
        for i in pred:
            cnt = 0
            for l in pred:
                
                if(i == l):
                    cnt += 1
            if(cnt > max):
                max = cnt
                predicted_class = i

        confidence = max/k * 100
        
        
        
        return  predicted_class, (y.argmax() == predicted_class), y.argmax(),confidence

   
        











def reverseImgShow(weights,num_classes):
        weights = [np.fliplr(x.get_value().transpose()) for x in weights]
        weights = weights[::-1]
        targets = np.diag(np.ones(num_classes))
        results = []
        for i in targets:
            temp = np.dot(i,weights[0])
            temp = np.dot(temp,weights[1])
            results.append(temp)
        results = [x.reshape((IMG_HEIGHT,IMG_WIDTH)).astype('float32') for x in results]
        return results




def gen_sets(data,targets,train,val,test):
    train,val,test = u.unit([train,val,test])
    
    training_set   = np.zeros((int(len(data)*train),2))
    validation_set = np.zeros((int(len(data)*val  ),2))
    test_set       = np.zeros((int(len(data)*test ),2))
    rng = np.random.RandomState(2)
    indices = np.arange(len(data))
    
    temp = rng.choice(indices,size=len(training_set),replace=False)
    training_indices = temp
    training_set = (np.vstack(data[temp]),np.vstack(targets[temp]))
    # training_set = (data[temp],targets[temp])
    indices = np.delete(indices,temp)
    
    temp = rng.choice(indices,size=len(validation_set),replace=False)
    validation_indices = temp
    validation_set = (np.vstack(data[temp]),np.vstack(targets[temp]))
    indices = np.delete(indices,temp)
    
    temp = rng.choice(indices,size=len(test_set),replace=False)
    testing_indices = temp
    test_set = (np.vstack(data[temp]),np.vstack(targets[temp]))
    indices = np.delete(indices,temp)
    
    return training_set, validation_set, test_set, (training_indices,validation_indices,testing_indices)
    
def load_KNN(data,targets,indices=None):
    k = KNN(data,targets)
    k.initD2(filename="100x100.npy",indices=indices)
    k_arr = np.arange(1,int(len(data)/2),2,dtype="int32")
    # k_values = k.test(k_arr)
    k_values = np.load("k_values.npy")
    k_sorted = sorted(k_values,key=lambda x:x[1])[::-1]
    best_k = k_sorted[0][0]
    return k, k_values, int(best_k)




def testKNN(training_set,validation_set,test):
    #  KNN has no validation step, so the training and validation sets are combined
    kdata = np.concatenate((training_set[0],validation_set[0]))
    
    ktargets= np.concatenate((training_set[1],validation_set[1]))
    k_indices = np.concatenate((indices[0],indices[1]))
    
    
    # 15.7229m calculating K values
    
    
    
    start_time = timeit.default_timer()
    k, k_values, best_k = load_KNN(kdata,ktargets,indices=k_indices)
    
    
    end_time = timeit.default_timer()
    print("%.4fm calculating K values" % ((end_time-start_time)/60.))
    
    print(best_k)
    
    pred = np.zeros((len(test_set[0])))
    start_time = timeit.default_timer()
    correct = 0
    results = []
    for i in range(len(test_set[0])):
        a = k.run(test_set[0][i],test_set[1][i],best_k)
        results.append(a)
        pred[i] = a[0]
        if(a[1]):
            correct +=1
        print("%s %d/%d (%d) correct"%((a,),correct,i+1,len(test_set[0])))
    
    end_time = timeit.default_timer()
    print("%.4fm testing with optimal K" % ((end_time-start_time)/60.))
    
    k_conf = u.confusion_matrix(pred,ktargets,num_classes=5)
    
    return results,k_conf

IMG_WIDTH = 100
IMG_HEIGHT = 100

batch_size = 30
num_classes=5
learning_rate=0.001     
L1_rg = 0.00
L2_rg = 0.0001



dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_classes,))])
infile = 'filenames5.txt'
filedata = np.loadtxt(infile,dtype=dt)
filenames = [a.decode('UTF-8') for a in filedata['filename']]
filenames.sort(key=lambda x:x[-7:])



data,targets,suffixes = u.get_images(w=IMG_WIDTH,h=IMG_HEIGHT,file_list = filenames)




training_set, validation_set, test_set,indices = gen_sets(data,targets,50,25,25)
training_batches   =  len(training_set[0])//batch_size 
validation_batches =  len(validation_set[0])//batch_size
test_batches       =  len(test_set[0])//batch_size


x = T.dmatrix('x')
y = T.lvector('y')



learning_rate = 0.01
L2 = 0.1
L1 = 0
size = [50,100,250,1000]
# for s in size:
classifier = Multilayer_Perceptron(x,
                        (IMG_WIDTH*IMG_HEIGHT,10),
                        num_classes=5,
                        rng=np.random.RandomState(0)
                        )
classifier.test(
    learning_rate=learning_rate,
    L2_rg=L2,
    L1_rg= L1,
    n_epochs=2000)








# 
# start_time = timeit.default_timer()
# results = k.test2(k_arr)
# np.save("k_values10x10",results)
# end_time = timeit.default_timer()
# print("%.4fm calculating K values" % ((end_time-start_time)/60.))
# 
# 
# k_ascending = sorted(results,key=lambda x:x[1])[-1]
# 
# print("Top 3 K values: %d, %d, %d" %(k_ascending[0],k_ascending[1],k_ascending[2]))
# 



# plt.figure(1)
# plt.subplot(2,1,1)
# x = k_arr
# y = k_err
# plt.axis([min(x), max(x)+1, 0, 100])
# plt.xlabel("K")
# plt.ylabel("Error")
# plt.title("All values of K up to 1291")
# plt.plot(x,y)
# plt.subplot(2,1,2)
# x = k_arr[:25]
# y = k_err[:25]
# plt.plot(x,y)
# plt.axis([min(x), max(x), 0, 100])
# plt.ylabel("Error")
# plt.xlabel("K")
# plt.xticks(np.arange(min(x), max(x)+1, 10))
# plt.title("First 25 values of K")
# plt.show()


































