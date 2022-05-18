"""Contains arguments for command-line parsing."""
import argparse


def parse_args():
    """Parse main arguments for `test.py`."""
    parser = argparse.ArgumentParser()

    # parameter_learning soon hopefully
    parser.add_argument("--bayes", type=str, default="data_driven_probs",
                        choices=['data_driven_probs', 'manual_probs'],
                        help="select Bayes Net defining approach")

    return parser.parse_args()

