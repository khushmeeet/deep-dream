import argparse


parser = argparse.ArgumentParser(description='Deep dreaming using resnet152')
parser.add_argument('--iters', default=10, type=int, help='Number of iterations for training')
parser.add_argument('--img', help='Input image for deep dream')
parser.add_argument('--layer', default=5, type=int, help='Layer used for deep dreaming')