import torch
from PIL.Image import open

from config import cfg
from data.transforms import build_transforms
from modeling import build_model


def predict(img):
    model = build_model(cfg)
    weight = torch.load(cfg.DIR.BEST_MODEL + cfg.TEST.WEIGHT)
    device = cfg.MODEL.DEVICE
    model.to(device=device)
    model.load_state_dict(weight)

    # Transform
    transform = build_transforms(cfg, is_train=False)
    input = transform(img)
    input = input.cuda()
    # unsqueeze batch dimension, in case you are dealing with a single image
    input = input.unsqueeze(0)

    # Set model to eval
    model.eval()

    # Get prediction
    output = model(input)
    output = output.data.cpu().numpy().argmax()
    label = ['apple',
             'aquarium_fish',
             'baby',
             'bear',
             'beaver',
             'bed',
             'bee',
             'beetle',
             'bicycle',
             'bottle',
             'bowl',
             'boy',
             'bridge',
             'bus',
             'butterfly',
             'camel',
             'can',
             'castle',
             'caterpillar',
             'cattle',
             'chair',
             'chimpanzee',
             'clock',
             'cloud',
             'cockroach',
             'couch',
             'crab',
             'crocodile',
             'cup',
             'dinosaur',
             'dolphin',
             'elephant',
             'flatfish',
             'forest',
             'fox',
             'girl',
             'hamster',
             'house',
             'kangaroo',
             'keyboard',
             'lamp',
             'lawn_mower',
             'leopard',
             'lion',
             'lizard',
             'lobster',
             'man',
             'maple_tree',
             'motorcycle',
             'mountain',
             'mouse',
             'mushroom',
             'oak_tree',
             'orange',
             'orchid',
             'otter',
             'palm_tree',
             'pear',
             'pickup_truck',
             'pine_tree',
             'plain',
             'plate',
             'poppy',
             'porcupine',
             'possum',
             'rabbit',
             'raccoon',
             'ray',
             'road',
             'rocket',
             'rose',
             'sea',
             'seal',
             'shark',
             'shrew',
             'skunk',
             'skyscraper',
             'snail',
             'snake',
             'spider',
             'squirrel',
             'streetcar',
             'sunflower',
             'sweet_pepper',
             'table',
             'tank',
             'telephone',
             'television',
             'tiger',
             'tractor',
             'train',
             'trout',
             'tulip',
             'turtle',
             'wardrobe',
             'whale',
             'willow_tree',
             'wolf',
             'woman',
             'worm']
    print(label[output])


if __name__ == '__main__':
    image = open('../data/dataset/predict/car2.jpeg')
    predict(img=image)
