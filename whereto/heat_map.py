import matplotlib.pyplot as plt

from .data_manipulation import manipulation


def plot_heatmap(R, title, soft_max=False):
    """plots a heatmap of the matrix R

    Args:
        R (numpy.ndarray): matrix to plot
        title (str, optional): title of the plot. Defaults to "".
    """
    plt.figure(figsize=(10, 10))
    if soft_max:
        R = manipulation(R)
    plt.imshow(R, cmap="hot")
    plt.title(title)
    plt.colorbar()
    plt.savefig(f"plots/{title}.png")
