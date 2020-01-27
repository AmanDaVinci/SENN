"""Main entry point of program. Here, a config file is parsed and the trainer is instantiated and run."""

import argparse
from utils.accuracy_vs_lambda import plot_lambda_accuracy
from trainer import init_trainer


def main():
    # get experiment configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/compas_config.json", help='experiment config file')
    args = parser.parse_args()

    trainer = init_trainer(args.config)
    plot_config, save_path, num_seed = trainer.run()
    trainer.finalize()
    _=plot_lambda_accuracy(plot_config, save_path, num_seed)


if __name__ == "__main__":
    main()
