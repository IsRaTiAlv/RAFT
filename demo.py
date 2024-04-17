import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import time

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder


# input_shape = (384, 512)
# input_shape = (768, 1024)
# input_shape = (1536, 2016)
# input_shape = (3040, 4032)
DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

def load_model(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    return model

def demo(args):
    model = load_model(args)
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=args.iterations, test_mode=True)
            viz(image1, flow_up)
    return 0
def profile(args):
    model = load_model(args)
    input_shape = tuple(args.dummy_input)
    times = []
    with torch.no_grad():
        for _ in range(100):
            image1 = torch.randn(1, 3, input_shape[0], input_shape[1], dtype=torch.float).to(DEVICE)
            image2 = torch.randn(1, 3, input_shape[0], input_shape[1], dtype=torch.float).to(DEVICE)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            start_time = time.time()
            _, _ = model(image1, image2, iters=args.iterations, test_mode=True)
            inference_time = (time.time() - start_time)

            times.append(inference_time)
    print(f"Elapsed time {np.mean(np.array(times)[1:])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="restore checkpoint")
    parser.add_argument('--iterations', required=False, default=20, help="restore checkpoint")
    parser.add_argument('--path', required=True, help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--profile', required=False, default=False, help='time_profiling')
    parser.add_argument('--dummy_input', required=False, nargs='+', type=int, help='time_profiling')

    args = parser.parse_args()
    if args.profile:
        profile(args)
    else:
        demo(args)
