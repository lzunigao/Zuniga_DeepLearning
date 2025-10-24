e = 2.71828
def ReLU(x):
    if x<0:
        return 0
    else:
        return x
    
def ReLU_derivative(x):
    if x<0:
        return 0
    else:
        return 1

def tanh(x, return_derivative=False):    
    if not return_derivative:
        return (2 / (1 + e**(-2*x))) - 1
    else:
        t = tanh(x)
        return 1 - t**2
    
def sigmoid(x, return_derivative=False):
    if not return_derivative:
        return 1 / (1 + e**(-x))
    else:
        s = sigmoid(x)
        return s * (1 - s)


def linear_bounded(x, return_derivative=False): # las agrego yo
    if not return_derivative:
        if x<-1:
            return -1
        elif x>1:
            return 1
        else:
            return x
    else:
        if x<-1 or x>1:
            return 0
        else:
            return 1

    
def linear_unbounded(x, return_derivative=False): # las agrego yo
    if not return_derivative:
        return x
    else:
        return 1
