import numpy as np
from scipy import optimize

# sigmoid function
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))
    
def cost(theta, x, y, lam=1.0):
    x = np.mat(x)
    y = np.mat(y)
    theta = np.mat(theta)
    m = float(len(y))
    
    assert np.shape(x[0,:]) == np.shape(theta)
    assert np.shape(x[:,0]) == np.shape(y[:,0])
    
    hx = sig(x * theta.transpose())
    
    reg_term = (lam/(2.0 * m)) * sum(power(theta[1:,0], 2))
    J = (1.0 / m) * sum((-y).transpose() * log(hx) - (1-y).transpose() * log(1-hx))
    J += reg_term
    
    return J
    
def gradient(theta, x, y, lam=1.0):
    x = np.mat(x)
    y = np.mat(y)
    theta = np.mat(theta)
    m = float(len(y))
    
    #print "x: %r, %r" % np.shape(x)
    #print "y: %r, %r" % np.shape(y)
    #print "theta: %r, %r" % np.shape(theta)
    
    assert np.shape(x[0,:]) == np.shape(theta)
    assert np.shape(x[:,0]) == np.shape(y[:,0])
    
    hx = sig(x * theta.transpose())
    grad = (1.0/m) * (hx - y).transpose() * x + ((lam/m) * theta)
    grad[0,0] -= ((lam/m) * theta[0,0])
    grad = ndarray.flatten(grad.A)
    return grad
    
class LR:
    def __init__(self, n_inputs, n_classes, lam=1):
        self.theta = np.mat(zeros((n_classes, n_inputs+1)))
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.lam = lam
        #print "initial theta: %r %r" % np.shape(self.theta)
        
    def train(self, x_vals, y, iters = 100, msg = True):
        x = np.mat(x_vals)
        # adding bias
        x0 = ones((np.shape(x[:,0])))
        x = concatenate((x0, x), 1)
        # train each class individually
        for i in xrange(self.n_classes):
            y_i = np.mat(y)[:,i]
            c = lambda t : cost(t, x, y_i, self.lam)
            g = lambda t : gradient(t, x, y_i, self.lam)
        
            t = self.theta[i,:]
            t_o = optimize.fmin_bfgs(c, t, g, disp = msg, maxiter=iters)
            self.theta[i,:] = t_o
            
    def hypothesis(self, x_vals):
        x = np.mat(x_vals)
        # adding bias
        x0 = ones((np.shape(x[:,0])))
        x = concatenate((x0, x), 1)
        return sig(x * self.theta.transpose())

    def predictions(self, x):
        hyp = self.hypothesis(x)
        return hyp.max(1) == hyp