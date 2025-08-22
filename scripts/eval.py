import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trust', action='store_true')
args = parser.parse_args()
print('Evaluation script placeholder', 'trust' if args.trust else '')
