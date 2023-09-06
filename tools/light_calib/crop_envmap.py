import numpy as np
import matplotlib.pyplot as plt
import imageio


def main():
    envmap = imageio.imread_v2("NC1_envmap.jpg")
    plt.imshow(envmap)
    plt.show()
    for i in range(envmap.shape[0]):
        if np.all(envmap[i] == 255):
            continue
        else:
            break
    for j in range(envmap.shape[0] - 1, 0, -1):
        if np.all(envmap[j] == 255):
            continue
        else:
            break
    plt.imshow(envmap[i:j])
    plt.show()
    imageio.imsave("/NC1_envmap_crop.jpg", envmap[i:j])


if __name__ == '__main__':
    main()
