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
import warnings
warnings.filterwarnings("ignore")
import sys
IMG_HEIGHT = 40
IMG_WIDTH = 40
num_labels = 5
num_images = 1291
layer1_size = 10
layer2_size = 5

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




# IMPORTING IMAGES INTO img_arr AND filedata

infile = 'filenames5.txt'
folder = 'tiffs5/'
abspath = 'C:/Users/Roan Song/Desktop/thesis/'


dt = np.dtype([('filename','|S16'),('labels',np.int32,(num_labels,))])
filedata = np.loadtxt(infile,dtype=dt)
filedata['labels'][filedata['labels'] == 0] = 0 # set to 0 if not using tanh


img_arr = np.zeros((num_images,IMG_HEIGHT*IMG_WIDTH))
labels_arr = np.copy(filedata['labels'])

for n in range(num_images):
    

    fname = filedata['filename'][n].decode('UTF-8')
    img = mpimg.imread(abspath + folder + fname)
    IN_HEIGHT = img.shape[0]
    IN_WIDTH = img.shape[1]
    
    img = pad_img(img,IMG_HEIGHT,IMG_WIDTH,IN_HEIGHT,IN_WIDTH)
    oneD = img.reshape(IMG_HEIGHT * IMG_WIDTH)
    img_arr[n] = oneD







# THEANO TIME BOIZ

def layer(X,w,bias=0.1):
    # b = np.array([bias], dtype=theano.config.floatX)
    # new_X = T.concatenate([X,b])
    # m = T.dot(w, new_X)
    m = T.dot(w.T, X)
    m += bias
    output = T.nnet.sigmoid(m)
    # output = np.tanh(m)
    return output


def grad_desc(cost, theta):
    eps = np.random.uniform(-variance,variance)
    return theta - (learning_rate*(1 + eps) * T.grad(cost, wrt=theta))



c = T.dvector('c')
x = T.dvector('x')
y = T.dvector('y')
w = T.dmatrix('w')
output = T.dvector('output')



x_arr = T.dmatrix('x_arr')
y_arr = T.dmatrix('y_arr')



num_pixels = IMG_HEIGHT*IMG_WIDTH


l1_weights  = theano.shared(np.array(np.random.rand(num_pixels,layer1_size) , dtype=theano.config.floatX))
l2_weights  = theano.shared(np.array(np.random.rand(layer1_size,layer2_size), dtype=theano.config.floatX))
out_weights = theano.shared(np.array(np.random.rand(layer2_size,num_labels) , dtype=theano.config.floatX))
pred        = theano.shared(np.array(np.zeros(num_labels), dtype=theano.config.floatX))


show_weights = theano.function([],[l1_weights, l2_weights, out_weights])

def export_weights():
    
    
    
    l1, l2, out = show_weights()
    
    l1  = np.array(l1)
    l2  = np.array(l2)
    out = np.array(out)
    
    
    
    return l1, l2, out
    


learning_rate = 0.01
variance = 0.5


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



cur_cost = 0
cost_arr = []


# confusion_total = np.array(np.zeros((iterations,num_labels,num_labels)),dtype='int32')
confusion_total = []
confusion_matrix = np.zeros((num_labels,num_labels))


def progress_bar(current,total):
    div = current/total
    bar_length = 20
    percentage = 100*div
    progress = '#'*int(bar_length*div) + ' '*(bar_length-int(bar_length*div))
    out_str = "Progress: [%s] %.1f%%\r" % (progress,percentage)
    sys.stdout.write(out_str)
    sys.stdout.flush()
    if(current >= total):
        print("Complete.")
        return

def run_straight(iterations):
    for i in range(iterations):
        for k in range(len(inputs)):
            feed_forward_update(inputs[k],y_pred[k])
    
        
    # winsound.Beep(440,250)

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
    # winsound.Beep(440,250)
 
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
    


cost_tuples = []
results_array = []



def update_results():
    correct, cost, conf = test()
    results_array.append(correct)
    cost_tuples.append(cost)
    details = [
    "Number correct: %s/%s" % (correct,num_images),
    "Average cost: %.4f" % (cost[0]),
    "Max cost: %.4f" % (cost[1]),
    "Min cost: %.4f" % (cost[2]),
    "Confusion Matrix:\n%s\n" % (conf)
    ]
    
    return '\n'.join(details)




plt.ion()

iter = 0
tfreq = 1
iter_tot = 0
while True:
    
    print()
    print(update_results())
    print()
    choice = input("1) run straight\n2) run batches\n3) plot\n4) learning rate\n5) variance\n6) plot2\n(Q)uit\n")
    
    if(choice == 'q' or choice == 'Q'):
        break
    
    elif(choice == '1' or choice == '2'):
        tfreq = eval(input("Test frequency:\n"))
        iter = eval(input("Number of iterations:\n"))
        iter_tot += iter
        if(choice == '2'):
            batch = eval(input("Batch Size:\n"))
            if(tfreq == 0):
                run_batches(iter,batch)
            else:
                for i in range(iter//tfreq):
                    run_batches(tfreq,batch)
                    update_results()
                    progress_bar(i+1,iter//tfreq)
        else:
            if(tfreq == 0):
                run_straight(iter)
            else:
                for i in range(iter//tfreq):
                    run_straight(tfreq)
                    update_results()
                    progress_bar(i+1,iter//tfreq)
        winsound.Beep(440,250)
        winsound.Beep(880,500)

    elif(choice == '3' or choice == 'p'):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        
        x_cost = np.linspace(0,iter_tot,num=len(cost_tuples))
        x_results = np.linspace(0,iter_tot,num=len(results_array))
        for temp in zip(*cost_tuples):
            ax1.plot(x_cost,np.array(temp))
        
        
        ax2.plot(x_results,results_array)
        plt.show(fig)
        plt.waitforbuttonpress()
    
    elif(choice == '4' or choice == 'l'):
        learning_rate = input("Enter new learning_rate\n")
    elif(choice =='5'):
        variance = input("Enter new variance\n")
    elif(choice =='6'):
        quick = []
        for o in range(len(inputs)):
            quick.append(feed_forward(inputs[o],y_pred[o]))
        
        fig = plt.figure()
        ax3 = fig.add_subplot(1,1,1)
        x_quick = range(len(inputs))
        for temp in zip(*quick):
            ax3.plot(x_quick,temp) 
        
        plt.show(fig)
        plt.waitforbuttonpress()   
     
    
# fig = plt.figure()
# fig.add_subplot(211)
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,1,2)

# for temp in zip(*cost_arr):
#      ax1.plot(x_raw,np.array(temp))
# 
# for temp in zip(*cost_tuples):
#      ax2.plot(x_tuples,np.array(temp))
# 
# ax1.set_title('Cost after every 100th batch')
# ax1.set_xlabel('Batch number (x100)')
# ax1.set_ylabel('Cost')
# ax2.set_ylabel('Cost')3
# ax2.set_xlabel('Epoch (10,000 iterations)')
# ax2.set_title('Cost per epoch')
# ax2.legend(['Average','Max','Min'],loc='upper right')
# ax1.legend(['Average','Max','Min'],loc='upper right')
# plt.show(fig)
# extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('test.pdf',bbox_inches=extent)
# fig.savefig('test.pdf',bbox_inches=extent.expanded(1.1,1.2))

        