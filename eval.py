import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from model import model
import torch.nn.functional as F

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ]
)

testmnist = datasets.MNIST('~/Downloads',train=False,transform=trans)

test_loader = DataLoader(testmnist,batch_size=1)

stored = torch.load('model.pt')
model.load_state_dict(stored.state_dict())
model.eval()
score = 0
with torch.no_grad():
    for x,y in test_loader:
        p = model(x.view(-1,28*28))
        p = F.softmax(p,dim=1)
        if torch.argmax(p, dim=1) == y:
            score+=1

print(100*(score/len(testmnist)))
