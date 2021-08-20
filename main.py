import torch
import torch.nn as nn
from torch.optim import SGD

from model import model
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ]
)

trainmnist = datasets.MNIST('~/Downloads',train=True,transform=trans)

train_loader = DataLoader(trainmnist,batch_size=64)

epochs = 10
lr = 0.005
stored = torch.load('model-.pt')
model.load_state_dict(stored.state_dict())
loss_fn = nn.CrossEntropyLoss()
optim = SGD(model.parameters(),lr=lr)
print('Started training!')
for e in range(epochs):
    for x,y in train_loader:
        x = x.view(-1,28*28)
        p = model(x)
        loss = loss_fn(p,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    print(loss)
    torch.save(model, f'model-{e}{loss}.pt')

print('Done training!')
print('Saved weights')
