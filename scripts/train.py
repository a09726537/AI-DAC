import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
args = parser.parse_args()
print(f'Training script placeholder, mode={args.mode}')
