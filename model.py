import torch.nn as nn
model = nn.Sequential(
    nn.Linear(28*28,128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10),

)
