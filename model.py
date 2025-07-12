import torch as t
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self,input_layer,hidden_layer,output_layer):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_layer,hidden_layer,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
       
            nn.Conv2d(hidden_layer,hidden_layer,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
     
            nn.Conv2d(hidden_layer,hidden_layer,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
       
            nn.Flatten(),
            nn.LazyLinear(hidden_layer//2),
            nn.ReLU(),
            nn.LazyLinear(hidden_layer//2,output_layer)
        )
    def forward(self,x):
        return self.net(x)