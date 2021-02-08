import sys
import torch
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
from models.pl_models import LigAdaIN
from models.AdaIN import mu, sigma


def loadImage(filename):
    input_image = Image.open(filename).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0)


if __name__ == "__main__":
    # declare variables to store hooks even though they aren't nessary for inference you need them to load the model from file
    model = LigAdaIN.load_from_checkpoint('lightning_logs/version_7/checkpoints/epoch=815-step=136180.ckpt')
    content_img = loadImage("datasets/img/content/C01001459133.jpg")
    style_img = loadImage("datasets/img/style/C01002748887.jpg")

    output_img = model(content_img, style_img)
    save_image(output_img, 'output.jpg')
