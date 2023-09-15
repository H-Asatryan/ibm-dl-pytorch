# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Quiz: Softmax Function: Using Lines to Classify Data
# (simple test commands)

import torch

# + [markdown] heading_collapsed=true
# # Task 1

# + hidden=true



# + hidden=true
 z = torch.tensor([[2,5,0],[10,8,2],[6,5,1]])
_, yhat = z.max(1) 

# + hidden=true
z.max(1)

# + hidden=true
yhat

# + [markdown] heading_collapsed=true
# # Task 2

# + hidden=true
 z = torch.tensor([[10,5,0],[10,8,2],[10,5,1]])

_, yhat = z.max(1)

yhat
# -

# # Task 3
#
# We have two input features and four classes, what are the parameters for Softmax() constructor according to the above code?

 class Softmax (torch.nn.Module):
        def __init__(self, in_size, out_size):
            super(Softmax, self).__init__()
            self.linear=torch.nn.Linear(in_size, out_size)
        
        def forward(self, x):
            out=self.linear(x)
            return out 

Softmax(2,4)

# # Task 4
#
# If we have two input features and three classes, what are the parameters for Softmax() constructor according to the above code?

Softmax(2,3)
