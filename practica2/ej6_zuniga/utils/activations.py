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

