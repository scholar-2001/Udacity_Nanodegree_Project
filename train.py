import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import argparse


parser = argparse.ArgumentParser(
    description = 'Argparsing for train.py'
)
parser.add_argument('data_dir', action='store', default='./flowers',nargs = '?')
parser.add_argument('--save_dir', action='store', default='./checkpoint.pth')
parser.add_argument('--arch', action='store', default='densenet121', choices = ('densenet121','vgg16'))
parser.add_argument('--learning_rate', action='store', type=float,default=0.003)
parser.add_argument('--hidden_units', action='store', type=int, default=512)
parser.add_argument('--epochs', action='store', type=int, default=5)
parser.add_argument('--dropout', action='store', type=float, default=0.4)
parser.add_argument('--gpu', action='store', default="gpu")

args = parser.parse_args()
data_dir = args.data_dir
chckpt_path = args.save_dir
l_r = args.learning_rate
trnsfr_model = args.arch
hidden_units = args.hidden_units
device_type = args.gpu
epochs = args.epochs
dropout = args.dropout


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ColorJitter(brightness = 0.5),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

exec('model = models.' + trnsfr_model + '(pretrained=True)')
for param in model.parameters():
    param.requires_grad = False


classifier = nn.Sequential(nn.Linear(1024 if trnsfr_model == 'densenet121' else 25088, hidden_units),
                          nn.ReLU(),
                          nn.Dropout(p=dropout),
                          nn.Linear(hidden_units, 102),
                          nn.LogSoftmax(dim=1)
                          )
    
model.classifier = classifier

device = torch.device("cuda" if torch.cuda.is_available() and device_type == 'gpu' else "cpu")
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=l_r)

model = model.to(device)

steps = 0
running_loss = 0
print_every = 20
for epoch in range(epochs):
    for inputs, labels in iter(trainloader):
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()


                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

                
model.class_to_idx = train_data.class_to_idx
model.cpu()
checkpoint = {'arch': trnsfr_model,
            'output_size': 102,
            'learning rate': l_r,
            'hidden units' : hidden_units,
            'dropout' : dropout,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx,
            'epochs' : epochs}
torch.save(checkpoint, chckpt_path)
print("Saved checkpoint!")