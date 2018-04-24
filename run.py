from settings import model_settings
from PIL import Image, ImageFilter, ImageChops
from utils import feature_map_at_layer
from torchvision import models
import matplotlib.pyplot as plt
import args
import utils


def deep_dream(model, image_path, layer, iters, lr, octave_scale, num_octaves):
    image = utils.load_image(image_path)
    if num_octaves > 0:
        img = image.filter(ImageFilter.GaussianBlur(2))
        if img.size[0] / octave_scale < 1 or img.size[1] / octave_scale < 1:
            size = img.size
        else:
            size = (int(img.size[0] / octave_scale), int(img.size[1] / octave_scale))
        
        img = img.resize(size, Image.ANTIALIAS)
        img = deep_dream(model, img, layer, iters, lr, octave_scale, num_octaves - 1)
        size = (image.size[0], image.size[1])
        img = img.resize(size, Image.ANTIALIAS)
        image = ImageChops.blend(image, img, 0.6)
    img_result = feature_map_at_layer(model, image, layer, lr, iters)
    img_result = img_result.resize(image.size)
    plt.imshow(img_result)
    return img_result

if __name__ == '__main__':
    options = args.parser.parse_args()
    resnet = models.resnet152(pretrained=True)
    deep_dream(resnet,
        options.img,
        options.layer,
        options.iters,
        model_settings['lr'],
        model_settings['octave_scale'],
        model_settings['num_octaves'])
            
