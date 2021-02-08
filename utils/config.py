import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Teeth Alignment')
    parser.add_argument("--config", type=str, default="config/default.yaml", help="path to config file")
    parser.add_argument("--gpus", type=str, default="0", help="the index of gpus, e.x. `0` or `0,1`, `-1` incidicating all gpus"
                                                              "and int(0) indicating cpu")

    args_cfg = parser.parse_args()
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg

def get_args(config):
    parser = argparse.ArgumentParser(description='Teeth Alignment')

    args_cfg = parser.parse_args()
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg
