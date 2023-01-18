import argparse

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=3407, type=int, help="seed")
    parser.add_argument("--entropy_constant", default=0.1, type=float, help="entropy constant")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num_episodes", default=50, type=int, help="total number of episodes")
    parser.add_argument("--num_epochs", default=100, type=int, help="epochs")
    parser.add_argument("--discount", default=0.1, type=float, help="discount")

    args = parser.parse_args()

    return args


def eval_args():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    return args