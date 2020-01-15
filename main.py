"""Main entry point of program. Here, a config file is parsed and the trainer is instantiated and run."""

import argparse
import json
from pprint import pprint
from types import SimpleNamespace

from trainer import Trainer


def main():
    # get experiment configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/compas_config.json", help='experiment config file')
    args = parser.parse_args()
    config_file = args.config

    with open(config_file, 'r') as f:
        config = json.load(f)
    print("==================================================")
    print(f" EXPERIMENT: {config['exp_name']}")
    print("==================================================")
    pprint(config)
    config = SimpleNamespace(**config)

    # create the trainer class and init with config
    trainer = Trainer(config)
    trainer.run()
    trainer.finalize()


if __name__ == "__main__":
    main()
