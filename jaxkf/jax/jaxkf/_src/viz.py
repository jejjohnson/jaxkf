import numpy as np
from matplotlib.patches import Ellipse, transforms

# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def plot_ellipse(
    Sigma,
    mu,
    ax,
    n_std=3.0,
    facecolor="none",
    edgecolor="k",
    plot_center="true",
    **kwargs
):
    cov = Sigma
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)

    if plot_center:
        ax.plot(mean_x, mean_y, ".")
    return ax.add_patch(ellipse)


def plot_tracking_values(observed, filtered, cov_hist, signal_label, ax):
    """
    observed: array(nsteps, 2)
        Array of observed values
    filtered: array(nsteps, state_size)
        Array of latent (hidden) values. We consider only the first
        two dimensions of the latent values
    cov_hist: array(nsteps, state_size, state_size)
        History of the retrieved (filtered) covariance matrices
    ax: matplotlib AxesSubplot
    """
    timesteps, _ = observed.shape
    ax.plot(
        observed[:, 0],
        observed[:, 1],
        marker="o",
        linewidth=0,
        markerfacecolor="none",
        markeredgewidth=2,
        markersize=8,
        label="observed",
        c="tab:green",
    )
    ax.plot(
        *filtered[:, :2].T, label=signal_label, c="tab:red", marker="x", linewidth=2
    )
    for t in range(0, timesteps, 1):
        covn = cov_hist[t][:2, :2]
        plot_ellipse(covn, filtered[t, :2], ax, n_std=2.0, plot_center=False)
    # ax.axis("equal")
    ax.legend()
    return ax
