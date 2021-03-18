#!/usr/bin/env python
# coding: utf-8

# In[75]:


# Import required libraries:
import numpy as np
# Define input features:
input_features = np.array([[0,0],[0,1],[1,0],[1,1]])
print (input_features.shape)
print (input_features)


# In[76]:


#OR 
# Define target output:
target_output = np.array([[0,1,1,1]]) 
# Reshaping our target output into vector
target_output = target_output.reshape(4,1)
print(target_output.shape)
print (target_output)


# In[77]:


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
plt.plot(target_output)


# In[78]:


# Define weights:
weights = np.array([[0.1],[0.2]])
print(weights.shape)
print (weights)
# Bias weight:
bias = 0.3
# Learning Rate:
lr = 0.05
# Sigmoid function:
def sigmoid(x):
    return 1/(1+np.exp(-x))
# Derivative of sigmoid function:
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


# In[79]:


# Main logic for neural network
for epoch in range(900):
    inputs = input_features#Feedforward input:
#Feedforward input:
    in_o = np.dot(inputs, weights) + bias #Feedforward output:
    out_o = sigmoid(in_o) #Backpropogation
#Calculating error
    error = out_o - target_output   
#Going with the formula:
    x = error.sum()
    print(x)
#Calculating derivative:
    derror_douto = error
    douto_dino = sigmoid_der(out_o)
    #Multiplying individual derivatives:
    deriv = derror_douto * douto_dino 
#Multiplying with the 3rd individual derivative:
#Finding the transpose of input_features:
    inputs = input_features.T
    deriv_final = np.dot(inputs,deriv)
    #Updating the weights values:
    weights -= lr * deriv_final #Updating the bias weight value:
    for i in deriv:
        bias -= lr * i #Check the final values for weight and biasprint (weights)


# In[80]:


print(weights)


# In[81]:


print(bias)


# In[82]:


#Taking inputs:
single_point = np.array([1,0]) 
#1st step:
result1 = np.dot(single_point, weights) + bias
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2) 
#Taking inputs:
single_point = np.array([1,1]) 

##EQUATION
#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2) 
#Taking inputs:
single_point = np.array([0,0]) 
#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2)
#Taking inputs:
single_point = np.array([0,1]) 
#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2)


# In[83]:


#AND
# Define target output:
target_output = np.array([[0,0,0,1]]) 
# Reshaping our target output into vector
target_output = target_output.reshape(4,1)
print(target_output.shape)
print (target_output)


# In[84]:


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
plt.plot(target_output)


# In[85]:


# Define weights:
weights = np.array([[0.1],[0.2]])
print(weights.shape)
print (weights)
# Bias weight:
bias = 0.4
# Learning Rate:
lr = 0.05
# Sigmoid function:
def sigmoid(x):
    return 1/(1+np.exp(-x))
# Derivative of sigmoid function:
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


# In[86]:


# Main logic for neural network
for epoch in range(900):
    inputs = input_features#Feedforward input:
#Feedforward input:
    in_o = np.dot(inputs, weights) + bias #Feedforward output:
    out_o = sigmoid(in_o) #Backpropogation
#Calculating error
    error = out_o - target_output   
#Going with the formula:
    x = error.sum()
    print(x)
#Calculating derivative:
    derror_douto = error
    douto_dino = sigmoid_der(out_o)
    #Multiplying individual derivatives:
    deriv = derror_douto * douto_dino 
#Multiplying with the 3rd individual derivative:
#Finding the transpose of input_features:
    inputs = input_features.T
    deriv_final = np.dot(inputs,deriv)
    #Updating the weights values:
    weights -= lr * deriv_final #Updating the bias weight value:
    for i in deriv:
        bias -= lr * i #Check the final values for weight and biasprint (weights)


# In[87]:


print (weights)


# In[88]:


print(bias)


# In[89]:



#Taking inputs:
single_point = np.array([1,0]) 
#1st step:
result1 = np.dot(single_point, weights) + bias
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2) 
#Taking inputs:
single_point = np.array([1,1]) 
#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2) 
#Taking inputs:
single_point = np.array([0,0]) 
#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2)
single_point = np.array([0,1]) 
#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:
result2 = sigmoid(result1) 
#Print final result
print(result2)


# In[90]:


#Sigmoid


# In[91]:


import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    return s,ds
x=np.arange(-6,6,0.01)
sigmoid(x)
# Setup centered axes
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# Create and show plot
ax.plot(x,sigmoid(x)[0], color="green, linewidth=3, label="sigmoid")
ax.plot(x,sigmoid(x)[1], color="yellow", linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
fig.show()


# In[92]:


import matplotlib.pyplot as plt
import numpy as np

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt
z=np.arange(-4,4,0.01)
tanh(z)[0].size,tanh(z)[1].size
# Setup centered axes
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# Create and show plot
ax.plot(z,tanh(z)[0], color="green", linewidth=3, label="tanh")
ax.plot(z,tanh(z)[1], color="yellow", linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
fig.show()


# In[93]:


def rectified(x):
    return max(0.0, x)


# In[94]:


# demonstrate with a positive input
x = 1.0
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
x = 1000.0
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
# demonstrate with a zero input
x = 0.0
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
# demonstrate with a negative input
x = -1.0
print('rectified(%.1f) is %.1f' % (x, rectified(x)))
x = -1000.0
print('rectified(%.1f) is %.1f' % (x, rectified(x)))


# In[95]:


# plot inputs and outputs
from matplotlib import pyplot
 
# rectified linear function
def rectified(x):
    return max(0.0, x)
 
# define a series of inputs
series_in = [x for x in range(-10, 11)]
# calculate outputs for our inputs
series_out = [rectified(x) for x in series_in]
# line plot of raw inputs to rectified outputs
pyplot.plot(series_in, series_out)
pyplot.show()


# In[96]:


#Superimpose


# In[97]:


import matplotlib.pyplot as plt
import numpy as np
def rectified(x):
    return max(0.0, x)
# define a series of inputs
series_in = [x for x in range(-10, 11)]
# calculate outputs for our inputs
series_out = [rectified(x) for x in series_in]
# line plot of raw inputs to rectified outputs

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt
def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    return s,ds
x=np.arange(-6,6,0.01)
sigmoid(x)
# Setup centered axes
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# Create and show plot
ax.plot(x,sigmoid(x)[0], color="black", linewidth=2, label="sigmoid")
ax.plot(x,tanh(x)[0], color="orange", linewidth=2, label="tanh")
ax.plot(series_in, series_out, color="r", linewidth=2, label="relu")
ax.legend(loc="upper right", frameon=False)
fig.show()


# In[ ]:





# In[ ]:




