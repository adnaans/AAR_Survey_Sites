import numpy as np
import matplotlib.pyplot as plt


class Neural_Network(object):
    def __init__(self, Lambda=0):
        #Define Hyperparameters
        self.inputLayerSize = 5
        self.outputLayerSize = 1
        self.hiddenLayerSize = 10

        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

        self.Lambda = Lambda

    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*np.sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1

        return dJdW1, dJdW2

    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        #Set perturbation vector
        perturb[p] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)

        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)

        #Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)

        #Return the value we changed to zero:
        perturb[p] = 0

    #Return Params to original value:
    N.setParams(paramsInitial)

    return numgrad

from scipy import optimize


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad

    def train(self, trainX, trainY, testX, testY):
    	#Make an internal variable for the callback function:
    	self.X = trainX
    	self.y = trainY
    	self.testX = testX
    	self.testY = testY
    	#Make empty list to store training costs:
    	self.J = []
    	self.testJ = []
    	params0 = self.N.getParams()
    	options = {'maxiter': 200, 'disp' : True}
    	_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)
    	self.N.setParams(_res.x)
    	self.optimizationResults = _res


#Training Data:
newstrainX = np.array(([1, 1,1,9,8], [9,4,2,9,9], [1,5,1,8,9], [1,1,5,9,8], [1,1,3,9,1], [1,1,2,9,1],[1,6,2,9,6],[1,1,1,9,1],[1,5,4,9,1],[1,1,1,9,1], [5,2,4,5,7], [1, 2, 2, 9,9]), dtype=float)
newstrainY = np.array(([1], [1], [1], [1], [1],[1],[1],[1],[1],[1], [0], [0]), dtype=float)

#Testing Data:
newstestX = np.array(([1, 2,3,4,5], [1,1,1,9,2], [1,3,5,8,1], [1,2,4,1, 9]), dtype=float)
newstestY = np.array(([0], [1], [1], [0]), dtype=float)

corptrainX = np.array(([1, 3,1,9,9], [1,7,7,8,9], [1,2,1,8,2], [1,1,8,8,6], [1,8,2,8,8],[1,1,2,8,6],[1,7,2,8,1],[1,7,7,8,9], [7,3,5,4,9], [4,2,6,5,2], [2,3,6,1,8]), dtype=float)
corptrainY = np.array(([1], [1], [1], [1], [1],[1],[1],[1], [0], [0], [0]), dtype=float)

#Testing Data:
corptestX = np.array(([1, 2,3,4,5], [1,1,1,9,2], [1,3,5,8,1], [1,2,4,1, 9]), dtype=float)
corptestY = np.array(([0], [1], [1], [0]), dtype=float)


newsNeural = Neural_Network(Lambda=0.0001)
newsTrainer = trainer(newsNeural)

newsTrainer.train(newstrainX, newstrainY, newstestX, newstestY)

corpNeural = Neural_Network(Lambda=0.0001)
corpTrainer = trainer(corpNeural)

corpTrainer.train(corptrainX, corptrainY, corptestX, corptestY)
'''
xxx = np.array(([1,2,1,9,8]), dtype=float)
print('For NBC')
print(NN.forward(xxx))
'''
typenet = raw_input("Enter in the site you are creating (for now only news or corporation) (0 for stop):")
typenet = typenet.lower()
while(typenet!="0"):
    confirm = False
    while(confirm == False):
        if(typenet == "corporation"):
            neuralNet = corpNeural
            confirm = True
        elif(typenet == "news"):
            neuralNet = newsNeural
            confirm = True
        else:
            typenet = raw_input("Sorry, but I didn't get that. Only corporation and news is supported right now. Please enter again:")
            typenet = typenet.lower()

    bgcolor = int(raw_input("Please input the background color number (White=1,Red=2,Orange=3,Yellow=4,Green=4,Blue=6,Purple=7,Gray=8,Black=9):"))
    fontone = int(raw_input("Please input the main font family (Helvetica=1, Arial=2, Roboto=3, Times=4, Georgia=5, Oswald=6, SansSerif=7, Verdana=8):"))
    fonttwo = int(raw_input("Please input the second main font family (Helvetica=1, Arial=2, Roboto=3, Times=4, Georgia=5, Oswald=6, SansSerif=7, Verdana=8):"))
    fontcolorone = int(raw_input("Please input the first main font color number (White=1,Red=2,Orange=3,Yellow=4,Green=4,Blue=6,Purple=7,Gray=8,Black=9):"))
    fontcolortwo = int(raw_input("Please input the second main font color number (White=1,Red=2,Orange=3,Yellow=4,Green=4,Blue=6,Purple=7,Gray=8,Black=9):"))

    xxx = np.array(([bgcolor,fontone,fonttwo,fontcolorone,fontcolortwo]), dtype=float)
    print("If the value is less than 0.5, it is bad. If the value is greater than 0.5, it is ok/good.")
    print(neuralNet.forward(xxx))
    typenet = raw_input("Enter in the site you are creating (for now only news or corporation) (0 for stop):")
    typenet = typenet.lower()
