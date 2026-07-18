"""
Batch experiment runner — repeats train_test_time.py with incremental seeds.

Usage:
    python train_repeat.py -r 3 [all train_test_time.py args ...]

Each repetition resets all RNG seeds (random, numpy, torch) and appends
-rep{i} to the experiment name for wandb / checkpoint isolation.
"""

import sys
import subprocess
import argparse

REPEAT_HELP = """\
usage: python train_repeat.py -r N [train_test_time.py args ...]

Run train_test_time.py N times with seeds base, base+1, ..., base+N-1.

positional arguments:
  -r N, --repeat N    Number of repetitions (required)
  --base-seed S       Seed for first repetition (default: --seed value)

All other arguments are passed through to train_test_time.py.
Example:
  python train_repeat.py -r 3 --model DRNet --data-mode HT21 ^
      --training-mode unsupervised --pseudo-mode mixed ^
      --beta 0.001 --reg-mode l1 --prior-mean 0
"""


def main():
    argv = sys.argv[1:]

    if '-h' in argv or '--help' in argv:
        print(REPEAT_HELP)
        return

    # Extract -r N and optional --base-seed
    repeat = None
    base_seed = None
    passthrough = []
    i = 0
    while i < len(argv):
        if argv[i] in ('-r', '--repeat') and i + 1 < len(argv):
            repeat = int(argv[i + 1])
            i += 2
        elif argv[i] == '--base-seed' and i + 1 < len(argv):
            base_seed = int(argv[i + 1])
            i += 2
        else:
            passthrough.append(argv[i])
            i += 1

    if repeat is None:
        print('ERROR: -r / --repeat is required.', file=sys.stderr)
        print(REPEAT_HELP, file=sys.stderr)
        sys.exit(1)

    # Parse passthrough to extract --seed (default 42)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=42)
    parsed, _ = parser.parse_known_args(passthrough)
    base_seed = base_seed if base_seed is not None else parsed.seed

    for i in range(repeat):
        seed = base_seed + i
        # Append -rep{i} suffix to experiment name
        suffix = f'-rep{i}'
        cmd = [sys.executable, 'train_test_time.py', f'--seed={seed}',
               f'--experiment-suffix={suffix}'] + passthrough

        print('=' * 72)
        print(f'Repetition {i + 1} / {repeat}  |  seed={seed}')
        print(f'Command: {" ".join(cmd)}')
        print('=' * 72)

        ret = subprocess.run(cmd, shell=not sys.platform.startswith('linux'))
        if ret.returncode != 0:
            print(f'ERROR: repetition {i + 1} failed with exit code {ret.returncode}',
                  file=sys.stderr)
            sys.exit(ret.returncode)


if __name__ == '__main__':
    main()
