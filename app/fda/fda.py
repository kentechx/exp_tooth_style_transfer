import click, os, os.path as osp
import numpy as np
import cv2
from PIL import Image
from utils import show, show_switch, fda


def get_mask(im_content: np.ndarray):
    mask = im_content.max(-1) > 10
    tooth_mask = im_content[:, :, 1:].max(-1) > 128
    gum_mask = (im_content[:, :, 0] > 128) & (im_content[:, :, 1:].max(-1) < 128)
    return mask, tooth_mask, gum_mask


@click.command()
@click.option('--img_src', default='../../examples/style/C01002338680.jpg')
@click.option('--img_trg', default='../../examples/style/C01002157218.jpg')
@click.option('--content_dir', default='../../examples/content')
def run(img_src, img_trg, content_dir):
    im_src = cv2.imread(img_src)
    im_trg = cv2.imread(img_trg)
    content_src = cv2.imread(osp.join(content_dir, osp.basename(img_src)))
    content_trg = cv2.imread(osp.join(content_dir, osp.basename(img_src)))

    mask_src, tooth_mask_src, gum_mask_src = get_mask(content_src)
    mask_trg, tooth_mask_trg, gum_mask_trg = get_mask(content_trg)

    # tooth
    out_tooth = fda(im_src * tooth_mask_src[..., None], im_trg * tooth_mask_trg[..., None])
    out_tooth = out_tooth * tooth_mask_src[..., None]
    out_tooth = np.clip(out_tooth, 0, 255).astype('u1')
    show_switch(out_tooth, im_trg)

    # gum
    out_gum = fda(im_src * gum_mask_src[..., None], im_trg * gum_mask_trg[..., None])
    out_gum = out_gum * gum_mask_src[..., None]
    out_gum = np.clip(out_gum, 0, 255).astype('u1')
    show_switch(out_gum, im_trg)

    # out
    out = out_tooth * tooth_mask_src[..., None] + out_gum * gum_mask_src[..., None] * (1-tooth_mask_src[..., None])
    show_switch(out, im_trg)

    cv2.imwrite('out.png', out)


if __name__ == '__main__':
    run()