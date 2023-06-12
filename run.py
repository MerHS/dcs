import os.path
from argparse import ArgumentParser

from src.trainer import Trainer


def run(args):
    trainer = Trainer(args.batch_size, args.num_episode, args.save_path, args.device)
    if args.load_path is not None:
        trainer.load(args.load_path)

    trainer.train()


if __name__ == "__main__":
    this_path = os.path.dirname(os.path.abspath(__file__))

    args = ArgumentParser()
    args.add_argument("--num_episode", type=int, default=1000)
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--save_path", type=str, default=this_path)
    args.add_argument("--load_path", type=str, default=None)
    args.add_argument("--device", type=str, default="cpu")

    args = args.parse_args()
    print(args)

    run(args)
