from experiments_utils import run_maybe_remotely
from argparse import ArgumentParser


def g(x):
    return x



parser = ArgumentParser()

parser.add_argument("--use_slurm", type=bool, default=True)


if __name__ == "__main__":
    import sys
    __import__('pdb').set_trace()

    args = parser.parse_args()

    run_maybe_remotely(g, "f", "f_test", x=1, use_slurm=args.use_slurm)
