"""Main entry point of program. Here, a configs file is parsed and the trainer is instantiated and run."""

import argparse

from trainer import init_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/mnist_config.json", help='experiment config file')
    args = parser.parse_args()

    trainer = init_trainer(args.config)
    trainer.run()
    trainer.finalize()


if __name__ == "__main__":
    main()
