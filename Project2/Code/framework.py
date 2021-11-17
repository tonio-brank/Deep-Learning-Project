import math as m
import torch as t


#Besoin d'implementer NN.sequential,NN.Linear,NN.Relu,NN.tanh,NN.MSE,NN.backward,NN.forward,SGD
# example model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out),
# )


t.set_grad_enabled(False)

class NeuralNetwork(object):

    def __init__(self):
        self.operations = []

    class Linear():
        def __init__(self,in_features,out_features,bias=True):
            self.type = 1
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias
            if self.bias:
                self.bias = t.rand((1, in_features))
            else:
                self.bias = t.zeros((1, in_features))
            self.weights = t.rand((out_features,in_features))

        def evaluation(self,input):
            return t.matmul(input,self.weights) + self.bias

    def add(self,operation):
        self.operations.append(operation)

    def forward(self,input):
        for operation in self.operations:
            input = operation.evaluation(input)
        return input


nn = NeuralNetwork()
p = nn.Linear(2,3)
nn.add(p)
input=t.rand((5,3))
print(nn.forward(input))


# m = t.nn.Linear(20, 30)
# print(m)
# input = t.randn(128, 20)
# output = m(input)
# print(output.size())
# print(output)