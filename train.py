import argparse
import collections
import importlib
import numpy as np
import torch
from parse_config import ConfigParser



def main(config):

    SEED = config.config.get('seed', 1)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(SEED)

    Trainer = importlib.import_module('src.trainer.' 
                                      + config['trainer'].get('type', 'trainer'))
    trainer = Trainer.Trainer(config)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    config = ConfigParser.from_args(args)

    main(config)
