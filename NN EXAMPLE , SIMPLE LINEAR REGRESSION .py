#!/usr/bin/env python
# coding: utf-8

# # NN EXAMPLE , SIMPLE LINEAR REGRESSION 

# ## IMPORT THE LIBRERARIES 

# In[80]:


import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


# # GENERATE RANDOM DATA TO TRAIN ON 

# In[81]:


# WE WILL CREATE FACK DATA WITH LINEAR RELATIOSHIP 


# In[116]:


observations = 1000
xs=np.random.uniform(low=-10,high=10,size=(observations,1))
zs=np.random.uniform(-10,10,size=(observations,1))
inputs = np.column_stack((xs,zs))
print(inputs.shape)

we are about to create a Tow variable  linear models
f(x,z)=a*x + b*z + c , we will use the Numpy Randum method .uniform (LOW,HIGH,SIZE) np.random.uniform(low,high,size) that drew a number of random values bfrom an interval (low,high ) that have equal chances to be selected 
we will folow the classic structure of building the element in a 
SUPERVISED LEARNING MODEL 
1) INPUTS >>>>> done 
2) weight >>>>> computer
3) biases >>>>>computer 
4) outputs >>>>computer 
5 targets >>>>> to do 
# ## creat the targets we will aim at 

# In[83]:


#our target is defined by the formula
# f(x,y) = a*x +b*z +c + noise 
#the w1 = weight 

#w2 = weight 2  
#b = noise  we will declare it using randum formula yet we will be able to keep the underlying linear relathionshp 


# In[117]:


noise = np.random.uniform(-1,1,(observations,1))
targets = 2*xs - 3*zs + 5 + noise 
print(targets.shape)


# In[94]:


# Initialize the weights and biases with random values
weights = np.random.uniform(-1, 1, size=(2, 1))
biases = np.random.uniform(-1, 1, size=1)


# ## plot the training data 

# In[96]:


# In order to use the 3D plot, the objects should have a certain shape, so we reshape the targets.
# The proper method to use is reshape and takes as arguments the dimensions in which we want to fit the object.
targets = targets.reshape(observations, 1)
outputs = outputs.reshape(observations, 1)

# Plotting according to the conventional matplotlib.pyplot syntax

# Declare the figure
fig = plt.figure()

# A method allowing us to create the 3D plot
ax = fig.add_subplot(111, projection='3d')

# Use scatter function to plot the points
ax.scatter(inputs[:, 0], inputs[:, 1], targets, label='Targets')
ax.scatter(inputs[:, 0], inputs[:, 1], outputs, label='Outputs')

# Set labels
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# Set legend
ax.legend()

# You can fiddle with the azim parameter to plot the data from different angles. Just change the value of azim=100
# to azim = 0 ; azim = 200, or whatever. Check and see what happens.
ax.view_init(azim=100)

# So far we were just describing the plot. This method actually shows the plot. 
plt.show()

# We reshape the targets and outputs back to the shape that they were in before plotting.
# This reshaping is a side-effect of the 3D plot. Sorry for that.
targets = targets.reshape(observations,)
outputs = outputs.reshape(observations,)


# ### Initialize variables

# In[106]:


# We will initialize the weights and biases randomly in some small initial range.
# init_range is the variable that will measure that.
# You can play around with the initial range,
# High initial ranges may prevent the machine learning algorithm from learning.

init_range = 0.1

# IN THIS CASE OUR INITIAL BIASES AND WEIGHTS WILL BE PICKED RANDOMLY FROM THE INTERVAL [-0.1 , 0.1 ]
# Weights are of size k x m, where k is the number of input variables and m is the number of output variables
# In our case, the weights matrix is 2x1 since there are 2 inputs (x and z) and one output (y)

weights = np.random.uniform(low=-init_range, high=init_range, size=(2, 1))

# Biases are of size 1 since there is only 1 output. The bias is a scalar.

biases = np.random.uniform(low=-init_range, high=init_range, size=1)

#Print the weights to get a sense of how they were initialized.
print (weights)
print (biases)


# In[107]:


###  [[ 0.0596875 ] WEIGHT 1 
###  [-0.02602845]] WEIGHT 2 
###  BIASE 


# # Set a learning rate
# # WE MUST DECIDE A LEARNING RATE (ETA)
# Set some small learning rate (as we learned about  eta in the lecture). 
# 0.02 is going to work quite well for our example. 
#we can also try different values of ETA to check how different values os ETA affecteS THE SPEED OF OUR OPTIMIZATION'

learning_rate = 0.02
# In[108]:


# Set the learning rate and the number of iterations
learning_rate = 0.02
iterations = 100


# # TRAIN THE MODEL 

# In[112]:


# We iterate over our training dataset 100 times. That works well with a learning rate of 0.02.
# The proper number of iterations is something we will talk about later on, but generally
# a lower learning rate would need more iterations, while a higher learning rate would need less iterations
# keep in mind that a high learning rate may cause the loss to diverge to infinity, instead of converge to 0.

for i in range(100):
    
    # This is the linear model: y = xw + b equation
    
    outputs = np.dot(inputs, weights) + biases
    
    # FOR SIMPLICITY we declare the variable deltas that record the difference between the outputs and the targets 
    # Note that deltas here is a vector 1000 x 1
    
    deltas = outputs - targets
    
    # then we must calculate the loss 
    # We are considering the L2-norm loss, but divided by 2, so it is consistent with the lectures.
    # Moreover, we further divide it by the number of observations.
    # This is simple rescaling by a constant. We explained that this doesn't change the optimization logic,
    # as any function holding the basic property of being lower for better results, and higher for worse results
    # can be a loss function.
    
    loss = np.sum(deltas ** 2) / 2 / observations
    
    # We print the loss function value at each step so we can observe whether it is decreasing as desired.
    print(loss)
    
    # Another small trick is to scale the deltas the same way as the loss function
    # In this way our learning rate is independent of the number of samples (observations).
    # Again, this doesn't change anything in principle, it simply makes it easier to pick a single learning rate
    # that can remain the same if we change the number of training samples (observations).
    # You can try solving the problem without rescaling to see how that works for you.
    
    deltas_scaled = deltas / observations
    
    # Finally, we must apply the gradient descent update rules from the relevant lecture.
    # The weights are 2x1, learning rate is 1x1 (scalar), inputs are 1000x2, and deltas_scaled are 1000x1
    # We must transpose the inputs so that we get an allowed operation.
    
    # formula for the weight 
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
    
    # using the formula for the biases 
    biases = biases - learning_rate * np.sum(deltas_scaled)
    
    # The weights are updated in a linear algebraic way (a matrix minus another matrix)
    # The biases, however, are just a single number here, so we must transform the deltas into a scalar.
    # The two lines are both consistent with the gradient descent methodology. 


# In[114]:


# We print the weights and the biases, so we can see if they have converged to what we wanted.
# When declared the targets, following the f(x,z), we knew the weights should be 2 and -3, while the bias: 5.
print (weights, biases)

# Note that they may be convergING. So more iterations are needed.


# In[ ]:





# ### Plot last outputs vs targets
# Since they are the last ones at the end of the training, they represent the final model accuracy. <br/>
# The closer this plot is to a 45 degree line, the closer target and output values are.

# In[118]:


# Calculate the final outputs
final_outputs = np.dot(inputs, weights) + biases

# Plot the final outputs vs targets
plt.scatter(targets, final_outputs)
plt.xlabel('Targets')
plt.ylabel('Outputs')
plt.show()

We created a simple linear regression model to predict a numeric output based on a single input variable. Our dataset consisted of 1000 observations, where each observation had one input feature and one output label. We used the ordinary least squares method to find the best-fit line for the data.

To train the model, we first initialized the weights and biases randomly. We then iterated over the training dataset for 100 epochs and updated the weights and biases after each iteration using the gradient descent algorithm.

We used the L2-norm loss function to measure the difference between the predicted output and the true output. We calculated the loss at each iteration and printed it to monitor the model's training progress.

After training, we evaluated the model's performance by plotting the predicted outputs against the true outputs. The plot showed a 45-degree line, indicating that the predicted and true outputs were very close. This result suggested that our model was accurate and could be used for making predictions on new data.
# In[ ]:





# In[ ]:




