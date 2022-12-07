import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image
import json

parser = argparse.ArgumentParser(description = 'Argparsing for predict.py')

parser.add_argument('input',default = './flowers/test/12/image_04059.jpg',nargs = '?', action="store", type = str)
parser.add_argument('checkpoint',default = './checkpoint.pth', nargs = '?',action="store", type = str)
parser.add_argument('--top_k', default=1, action="store", type=int)
parser.add_argument('--category_names', action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store")

args = parser.parse_args()
imge_path = args.input
topk = args.top_k
curr_device = args.gpu
cat_file = args.category_names
chck_path = args.checkpoint

checkpoint = torch.load(chck_path)

exec('model = models.' + checkpoint['arch'] + '(pretrained=True)')

for param in model.parameters():
    param.requires_grad = False

model.class_to_idx = checkpoint['class_to_idx']

classifier = nn.Sequential(nn.Linear(1024 if checkpoint['arch'] == 'densenet121' else 25088, checkpoint['hidden units']),
                           nn.ReLU(),
                          nn.Dropout(p = checkpoint['dropout']),
                          nn.Linear(checkpoint['hidden units'], 102),
                          nn.LogSoftmax(dim=1)
                          )

model.classifier = classifier
model.load_state_dict(checkpoint['state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() and curr_device == "gpu"  else "cpu")
model.to(device)
def process_image(imge_path):
    imge = Image.open(imge_path)
    imge_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    imge_tensor = imge_transform(imge)
    return imge_tensor.to(device)
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f, strict=False)

def predict(imge_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    processed_image = process_image(imge_path)
    processed_image.unsqueeze_(0)
    model.eval()
    probs = torch.exp(model.forward(processed_image))
    top_probabs, top_labels = probs.topk(topk)

    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key
    new_top = top_labels.to('cpu')
    np_top_labels = new_top[0].numpy()

    top_labls = [int(idx_to_class[label]) for label in np_top_labels]

    
    return top_probabs, top_labls
prob, label = predict(imge_path, model, topk)
prob1 = prob.to('cpu')
prob2 = prob1[0].detach().numpy() 
for i in range(topk):
    print('It is a {} with a probability of {:3f}'.format(cat_to_name[str(label[i])], prob2[i]))
                                 