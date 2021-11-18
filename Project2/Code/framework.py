import math as m
import torch as t

#Besoin d'implementer NN.sequential,NN.Linear,NN.Relu,NN.tanh,NN.MSE,NN.backward,NN.forward,SGD

# example model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out),
# )

# t.set_grad_enabled(False)

class NeuralNetwork(object):

    def __init__(self):
        self.operations = []

    class Linear():
        def __init__(self,in_features,out_features,bias=True):
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias

            if self.bias:
                self.bias = t.tensor([1/in_features]) # pytorch initialize the bias like this, we will broadcast this value to obtain the correct shape to add it
            else:
                self.bias = t.tensor([0])

            self.weights = t.rand((in_features,out_features)) # input features and output features swapped to have the transposition

        def evaluation(self,input):
            mul = t.matmul(input,self.weights)
            self.bias = t.full(mul.size(),self.bias[0])
            return mul + self.bias

    def add(self,*operation):
        for op in operation:
            self.operations.append(op)

    class reLU():
        def evaluation(self,input):
            return input.apply_(lambda x: (max(0, x)))

    class tanH():
        def evaluation(self,input):
            return input.apply_(lambda x: (m.exp(x) - m.exp(-x)) / (m.exp(x) + m.exp(-x)))

    def forward(self,input):
        for operation in self.operations:
            input = operation.evaluation(input)
        return input

    def MSE(self,y_train,output):
        y_train = t.flatten(y_train).tolist()
        output = t.flatten(output).tolist()
        N = len(y_train)
        res = 0
        for x,y in zip(y_train,output):
            res += (x-y)**2
        return res/N

# Neural Network example FP
nn = NeuralNetwork()
nn.add(nn.Linear(5,2),nn.reLU(),nn.Linear(2,1),nn.tanH())
input=t.rand((10,5))

print(nn.forward(input))