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
@click.option('--real_dir', default='/mnt/share/shenfeihong/data/test_03_26')
@click.option('--sim_dir', default='/mnt/share/shenkaidi/MyProjects/Render/smile_render_result/out/C01004899864')
def run(real_dir, sim_dir):
    case_id = osp.basename(sim_dir)

    # read images from the real dir
    real_img = cv2.imread(osp.join(real_dir, case_id, 'mouth.jpg'))
    mouth_mask = cv2.imread(osp.join(real_dir, case_id, 'MouthMask.png')).max(-1) > 0
    teeth_mask = cv2.imread(osp.join(real_dir, case_id, 'TeethMasks.png')).max(-1) > 0
    # show_switch(mouth_mask, teeth_mask)
    teeth_real_img = real_img * teeth_mask[..., None]

    # read images from the sim dir
    sim_img = cv2.imread(osp.join(sim_dir, 'step3.jpg'))
    teeth_sim_img = sim_img * teeth_mask[..., None]

    # transfer teeth
    teeth_sim_to_real = fda(teeth_sim_img, teeth_real_img, L=0.05)
    teeth_real_to_sim = fda(teeth_real_img, teeth_sim_img, L=0.05)
    show_switch(teeth_sim_to_real, teeth_sim_img)
    # show_switch(teeth_sim_to_real, teeth_real_img)
    # show_switch(teeth_real_to_sim, teeth_sim_img)

    # out
    out = real_img * (1 - teeth_mask[..., None]) + teeth_sim_to_real * teeth_mask[..., None]
    show_switch(out, real_img)

    cv2.imwrite('out.png', out)


if __name__ == '__main__':
    run()
