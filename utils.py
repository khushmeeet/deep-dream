import PIL.Image
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.autograd import Variable
import torch
import numpy as np


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


def load_image(path):
    image = PIL.Image.open('./images/img_0.jpg')
    return image


def feature_map_at_layer(model, image, layer, lr, iterations, options):
    if options.cuda:
        img = Variable(preprocess(image).unsqueeze(0).cuda(), requires_grad=True)
    else:
        img = Variable(preprocess(image).unsqueeze(0), requires_grad=True)
    model.zero_grad()
    module_list = list(model.features.modules())
    for _ in range(iterations):
        out = img
        for j in range(layer):
            out = module_list[j + 1](out)
        loss = out.norm()
        loss.backward()
        img.data = img.data + (lr * img.grad.data)
    
    img = img.data.squeeze()
    img.transpose_(0, 1)
    img.transpose_(1, 2)
    img = np.clip(deprocess_image(img, options), 0, 1)
    img = PIL.Image.fromarray(np.uint8(img * 255))
    return img


def deprocess_image(image, options):
    if options.cuda:
        return image * torch.Tensor([0.229, 0.224, 0.225]).cuda() + torch.Tensor([0.485, 0.456, 0.406]).cuda()
    else:
        return image * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])
