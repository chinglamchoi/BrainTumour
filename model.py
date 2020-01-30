import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=7)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=7)
        #maxpool
        self.conv3 = nn.Conv2d(32, 64, kernel_size=7)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=7)
        #maxpool
        self.fc1 = nn.Linear(166464, 512) ##
        self.output = nn.Linear(512, 3) ##

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.leaky_relu(F.max_pool2d(out, kernel_size=3, stride=3))
        
        out = self.conv3(out)
        out = self.conv4(out)
        out = F.leaky_relu(F.max_pool2d(out, kernel_size=3, stride=3))
        
        out = out.view(out.size(0), -1)

        out = F.leaky_relu(self.fc1(out))
        out = F.leaky_relu(self.output(out))
        return out

def run_cnn():
    return ConvNet()
