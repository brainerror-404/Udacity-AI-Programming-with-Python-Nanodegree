import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from PIL import Image
from collections import OrderedDict
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as s

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--image_path', dest='image_path', default='flowers/test/12/image_03994.jpg')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def load_checkpoint(filepath='checkpoint.pth'):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(4096, 512)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.2)),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    checkpoint = torch.load(filepath)
    epochs = checkpoint['epochs']
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image):
    image = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    image = img_transforms(image)
    return image


def predict(image_path, model, topk=5, gpu='gpu'):
    img = process_image(image_path)
    img = img.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        if gpu == 'gpu':
            model = model.to('cuda')
            logps = model.forward(img.cuda())
        else:
            logps = model.forward(img)

        ps = torch.exp(logps)
        probs, indices = ps.topk(topk, dim=1)
        probs = probs.cpu().numpy()[0]
        indices = indices.cpu().numpy()[0]

        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[i] for i in indices]

        return probs, classes



def main():
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    image_path = args.image_path
    probs, classes = predict(image_path, model, int(args.top_k), gpu)
    flower_name = [cat_to_name[i] for i in classes]

    print(flower_name)
    print(probs)

    i = 0
    while i < len(flower_name):
        print("{} has a probability of {}".format(flower_name[i], probs[i]))
        i += 1

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()