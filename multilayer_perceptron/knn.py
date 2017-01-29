import numpy as np
import timeit
import utilities as u
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
            
                u.progress_bar(i,len(self.data))    
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
        
        
        
        
def load_KNN(data,targets,indices=np.array(False),filename=None):
    k = KNN(data,targets)
    if(indices.any()):
        k.initD2(filename=filename,indices=indices)
    elif(filename):
        k.D2 = np.load(filename=filename)
    else:
        k.initD2()
    
    print("Square distance matrix initialised")
    k_arr = np.arange(1,int(len(data)/8),2,dtype="int32")
    k_values = k.test(k_arr)
    # k_values = np.load("k_values_old.npy")
    k_sorted = sorted(k_values,key=lambda x:x[1])[::-1]
    best_k = k_sorted[0][0]
    print("Best value of K obtained")
    k.best_k = int(best_k)
    k.k_values = k_values
    print(k.D2.shape)
    print(best_k)
    return k




def testKNN(training_set,validation_set,test_set,indices,k=None,filename=None):
    #  KNN has no validation step, so the training and validation sets are combined
    kdata = np.concatenate((training_set[0],validation_set[0]))
    
    ktargets = np.concatenate((training_set[1],validation_set[1]))
    k_indices = np.concatenate((indices[0],indices[1]))
    
    
    # 15.7229m calculating K values
    # 473/528
    
    
    start_time = timeit.default_timer()
    classifier = load_KNN(kdata,ktargets,indices=k_indices,filename=filename)
    k_values = classifier.k_values
    if not k:
        k = classifer.best_k
    
    end_time = timeit.default_timer()
    print("%.4fm calculating K values" % ((end_time-start_time)/60.))
    
    print(k)
    
    pred = np.zeros((len(test_set[0])))
    start_time = timeit.default_timer()
    correct = 0
    results = []
    for i in range(len(test_set[0])):
        temp = classifier.run(test_set[0][i],test_set[1][i],k)
        results.append(temp)
        pred[i] = int(temp[0])
        if(temp[1]):
            correct +=1
        u.progress_bar(i,len(test_set[0]))
    print("%d/%d correct"%(correct,len(test_set[0])))
    
    end_time = timeit.default_timer()
    print("%.4fm testing with optimal K" % ((end_time-start_time)/60.))
    
    targets = test_set[1].argmax(axis=1)
    
    classifier.conf = u.confusion_matrix(pred,targets,num_classes=8)
    
    
    
    return classifier,results