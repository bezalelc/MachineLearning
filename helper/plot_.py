import numpy as np


def img_show_data(images: list[np.ndarray], labels: list[str], hspace=0.4, wspace=0.3) -> None:
    import matplotlib.pyplot as plt
    n = int(len(images) ** 0.5)
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_xlabel('Label : %s' % labels[i])
    plt.show()
