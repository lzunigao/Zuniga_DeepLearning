e = 2.71828
def ReLU(x):
    if x<0:
        return 0
    else:
        return x

def tanh(x):
    return (2 / (1 + e**(-2*x))) - 1

def sigmoid(x):
    return 1 / (1 + e**(-x))

def linear_bounded(x): # las agrego yo
    if x<-1:
        return -1
    elif x>1:
        return 1
    else:
        return x
    
def linear_unbounded(x):
    return x