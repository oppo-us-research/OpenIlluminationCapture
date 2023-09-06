import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outpath", default="")
    # parser.add_argument("--nframes", default=12)
    parser.add_argument("--sortn", default=4)
    parser.add_argument("--randns", nargs="+", default=[50, 60, 70, 80, 90])
    parser.add_argument("--intensity", default=31)

    args = parser.parse_args()

    nlight = 142
    sortn = args.sortn
    randns = args.randns
    nframes = len(randns) + sortn
    intensity = args.intensity

    a = np.zeros((200, nframes))
    for i in range(sortn):
        a[i * (nlight // sortn):(i + 1) * (nlight // sortn), i] = intensity
    for i in range(sortn, nframes):
        a[:nlight, i] = np.random.permutation([intensity] * randns[i-sortn] + [0] * (nlight - randns[i-sortn]))
    if args.outpath != "":
        np.savetxt(args.outpath, a, fmt="%d")
    else:
        print("No output path specified, printing to stdout")
        print(a)


if __name__ == '__main__':
    main()
