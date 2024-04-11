import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action='store', default='flowers/')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='5')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth')
    return parser.parse_args()

def train(model, data_dir, save_dir, learning_rate, hidden_units, epochs, gpu):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64)

    if model == 'vgg16':
        model = models.vgg16(pretrained=True)
        print("vgg16 is used...")
    elif model == 'densenet121':
        model = models.densenet121(pretrained=True)
        print("densenet121 is used...")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU is available: ", torch.cuda.is_available())
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(learning_rate))

    epochs = int(epochs)
    steps = 0
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for inputs, labels in trainloaders:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloaders:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Train loss: {running_loss / print_every:.3f}, "
                      f"Valid loss: {valid_loss / len(validloaders):.3f}, "
                      f"Valid accuracy: {accuracy / len(validloaders) * 100:.2f}%")

                running_loss = 0
                model.train()

    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    return model

def main():
    args = parse_args()
    train(args.arch, args.data_dir, args.save_dir, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
    print("Image Directory: ", args.data_dir, "\nSave Directory: ", args.save_dir, "\nModel: ", args.arch,
          "\nLearning Rate: ", args.learning_rate, "\nEpochs: ", args.epochs, "\nHidden Units: ", args.hidden_units)

if __name__ == "__main__":
    main()