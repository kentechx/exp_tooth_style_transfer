import click
import numpy as np
import cv2
from PIL import Image
from utils import FDA_source_to_target_np


@click.command()
@click.option('--img_src', default='../examples/style/C01002338680.jpg')
@click.option('--img_trg', default='../examples/style/C01002157218.jpg')
def run(img_src, img_trg):
    im_src = Image.open(img_src).convert('RGB')
    im_trg = Image.open(img_trg).convert('RGB')

    # im_src = im_src.resize((1024,512), Image.BICUBIC )
    # im_trg = im_trg.resize((1024,512), Image.BICUBIC )

    im_src = np.asarray(im_src, np.float32)
    im_trg = np.asarray(im_trg, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.01)

    src_in_trg = src_in_trg.transpose((1, 2, 0))
    src_in_trg = np.clip(src_in_trg, 0, 255).astype('u1')
    Image.fromarray(src_in_trg).save('out.png')


if __name__ == '__main__':
    run()