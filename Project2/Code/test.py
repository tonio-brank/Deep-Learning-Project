import framework as fw
import torch as t
import math as m


# Generating test set
print("Generate x_test....")
print("")
x_test = t.rand((1000,2))
radius = 1/((2*m.pi)**(1/2))
limit_up = 0.5+radius
limit_down = 0.5-radius
y_test = t.tensor([0 if ( (pair[0]>limit_up or pair[1]>limit_up) or (pair[0]<limit_down or pair[1]<limit_down)) else 1 for pair in x_test])

print("Some examples...")
print("")
print("radius: ",radius)
print("down limit: ",limit_down)
print("up limit: ",limit_up)
print("")
print("coordinate: ",x_test[0:4])
print("corresponding label ",y_test[0:4])



