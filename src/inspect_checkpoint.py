import sys

import torch


def main():
    checkpoint = torch.load(sys.argv[1], map_location=torch.device('cpu'))
    print(checkpoint.keys())


if __name__ == '__main__':
    main()
