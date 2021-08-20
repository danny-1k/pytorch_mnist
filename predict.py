import numpy as np
#import matplotlib.pyplot as plt
import PIL
import torch
from torchvision import transforms
from model import model
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(
    description='Number classifier trained on the MNIST dataset')
parser.add_argument('--model',help='trained model weights')
parser.add_argument('--img',help='image of a number to classify')
args = parser.parse_args()


trans = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ]
)

stored = torch.load(args.model)
model.load_state_dict(stored.state_dict())
model.eval()
x = PIL.Image.open(args.img)
x = np.asarray(x).reshape(28,28,3)
x = PIL.Image.fromarray(x)
x = trans(x)
with torch.no_grad():
    p = model(x.view(-1,28*28))

    p = F.softmax(p,dim=1)

    for num,i in enumerate(p.squeeze()):
        score = round(i.numpy().tolist()*100/10)
        print(f'{"#"*score}{" "*(10-score)} {num}')
