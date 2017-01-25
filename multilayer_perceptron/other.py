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
    elm_A = u.normalise(elm_A)
    elm_A = np.insert(elm_A,0,1,axis=1)
    elm_T = np.concatenate((elm_class_one_T,elm_class_two_T),axis=0)
    elm_X = np.array([[1,7],[2,8],[2,9],[10,1],[8,2],[9,1]])
    elm_X = u.normalise(elm_X)
    def __init__(self,A,t,X,lam=0.01):
        self.A = np.insert(u.normalise(A),0,1,axis=1)
        self.X = u.normalise(X)
        self.t = t
        self.R =  np.random.uniform(-1,1,(self.A.shape[1],t.shape[1]))
        self.w = np.dot(u.pinv(np.tanh(np.dot(self.A,self.R)),lam),t)
        self.h = np.dot(self.X,self.w)

def ELM(B,T,X,lam=0.001):
    A = np.insert(B,0,1,axis=1)
    R =  np.random.uniform(-1,1,(A.shape[1],T.shape[1]))
    Ar = np.tanh(np.dot(A,R))
    A_pinv = u.pinv(Ar,lam)
    w = np.dot(A_pinv,T)
    h = np.dot(X,w)
    return (X,softmax(h),np.argmax(h,axis=1))
    
# seek to minimize ||Xw - t||^2 (i.e. find least-squares solution)
# value of w which does this is A_pinv*T
# A = USV' A_pinv = VS^-1U'
# seek to minimize lambda||w||^2 as well
# i.e. (1/n)*||Xw - t||^2 + lam*||w||^2
# big lambda = small w but more innaccurate