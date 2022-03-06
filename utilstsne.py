from os.path import abspath, dirname, join

import numpy as np
import math
import scipy.sparse as sp
import matplotlib.pyplot as plt
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

FILE_DIR = dirname(abspath(__file__))
DATA_DIR = join(FILE_DIR, "data")


def unit_vector(vector):
    # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    # https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotate(points, rotation_point):
    # https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
    baseline_vector = np.array((-1.0, 1.0))
    theta = angle_between(baseline_vector, rotation_point)
    print(baseline_vector, rotation_point, theta)
    if points.shape[1] == 2:
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        points = np.dot(points, R.T)
        return(points)
    else:
        raise Exception('Rotation is only defined for 2D points.')


def log_normalize(data):
    """Perform log transform log(x + 1).

    Parameters
    ----------
    data : array_like

    """
    if sp.issparse(data):
        data = data.copy()
        data.data = np.log2(data.data + 1)
        return data

    return np.log2(data.astype(np.float64) + 1)


def pca(x, n_components=50):
    if sp.issparse(x):
        x = x.toarray()
    U, S, V = np.linalg.svd(x, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    x_reduced = np.dot(U, np.diag(S))
    x_reduced = x_reduced[:, np.argsort(S)[::-1]][:, :n_components]
    return x_reduced


def iteractive_plot(df, x_label, y_label, z_label=None, task='classification', sorted_classes = {}, color_label=[], colors=[], size=80, hover_name='index'):
    df['index'] = df.index
    dot_size = np.empty(df.shape[0])
    dot_size.fill(size)
    if task == 'classification':
        print('sorted_classes')
        print(sorted_classes)
        if z_label is None:
            fig = px.scatter(df, x=x_label, y=y_label, category_orders=sorted_classes, color=color_label, color_discrete_sequence=colors,
                         symbol=color_label, size=dot_size, hover_name=hover_name)
        else:
            fig = px.scatter_3d(df, x=x_label, y=y_label, z=z_label, category_orders=sorted_classes, color=color_label, color_discrete_sequence=colors,
                         symbol=color_label, size=dot_size, hover_name=hover_name)        
    elif task == 'regression':
        df[color_label] = df[color_label].astype(float)
        if z_label is None:
            fig = px.scatter(df, x=x_label, y=y_label, color=color_label, color_continuous_scale='balance', size=dot_size, hover_name=hover_name)
        else:
            fig = px.scatter_3d(df, x=x_label, y=y_label, z=z_label, color=color_label, color_continuous_scale='balance', size=dot_size, hover_name=hover_name)         
    return fig

def plot(
    x,
    y,
    task='classification',
    class_label=None,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):
    import matplotlib

    n_dimensions = x.shape[1]
    if n_dimensions > 3:
        n_dimensions = 3
        print('Number of dimensions is larger than 3, plotting the first 3 components.')
    #if n_dimensions == 3:
        #fig = plt.figure()
        #ax = Axes3D(fig)

    if ax is None:
        if n_dimensions < 3:
            _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))
        else:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}



    if task == 'classification':

        # Create main plot
        if label_order is not None:
            assert all(np.isin(np.unique(y), label_order))
            classes = [l for l in label_order if l in np.unique(y)]
        else:
            classes = np.unique(y)
        if colors is None:
            default_colors = matplotlib.rcParams["axes.prop_cycle"]
            colors = {k: v["color"] for k, v in zip(classes, default_colors())}

        point_colors = list(map(colors.get, y))

        if n_dimensions == 1:
            ax.scatter(x[:, 0], x[:, 0], c=point_colors, rasterized=True, **plot_params)
        elif n_dimensions == 2:
            ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)
        else:
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=point_colors, rasterized=True, **plot_params)

        # Plot mediods
        if draw_centers:
            centers = []
            for yi in classes:
                mask = yi == y
                if n_dimensions < 3:
                    centers.append(np.median(x[mask, :2], axis=0))
                else:
                    centers.append(np.median(x[mask, :3], axis=0))
            centers = np.array(centers)

            center_colors = list(map(colors.get, classes))
            if n_dimensions == 1:
                ax.scatter(
                    centers[:, 0], centers[:, 0], c=center_colors, s=48, alpha=1, edgecolor="k", marker="s"
                )
            elif n_dimensions == 2:
                ax.scatter(
                    centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k", marker="s"
                )
            else:
                 ax.scatter(
                    centers[:, 0], centers[:, 1], centers[:, 2], c=center_colors, s=48, alpha=1, edgecolor="k", marker="s"
                )               

            # Draw mediod labels
            if draw_cluster_labels:
                for idx, label in enumerate(classes):
                    ax.text(
                        centers[idx, 0],
                        centers[idx, 1] + 1.5,
                        label,
                        fontsize=kwargs.get("fontsize", 6),
                        horizontalalignment="center",
                    )
    
    elif task == 'regression':

        target_color = y.astype(float)
        if n_dimensions == 1:
            s = ax.scatter(x[:, 0], x[:, 0], c=target_color, cmap='coolwarm')
        elif n_dimensions == 2:
            s = ax.scatter(x[:, 0], x[:, 1], c=target_color, cmap='coolwarm')
        else:
            s = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=target_color, cmap='coolwarm')
        cbar = plt.colorbar(s, ax=ax)
        cbar.set_label(class_label)
        draw_legend = False

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=15,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=24, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)
    return ax


def evaluate_embedding(
    embedding, labels, projection_embedding=None, projection_labels=None, sample=None
):
    """Evaluate the embedding using Moran's I index.

    Parameters
    ----------
    embedding: np.ndarray
        The data embedding.
    labels: np.ndarray
        A 1d numpy array containing the labels of each point.
    projection_embedding: Optional[np.ndarray]
        If this is given, the score will relate to how well the projection fits
        the embedding.
    projection_labels: Optional[np.ndarray]
        A 1d numpy array containing the labels of each projection point.
    sample: Optional[int]
        If this is specified, the score will be computed on a sample of points.

    Returns
    -------
    float
        Moran's I index.

    """
    has_projection = projection_embedding is not None
    if projection_embedding is None:
        projection_embedding = embedding
        if projection_labels is not None:
            raise ValueError(
                "If `projection_embedding` is None then `projection_labels make no sense`"
            )
        projection_labels = labels

    if embedding.shape[0] != labels.shape[0]:
        raise ValueError("The shape of the embedding and labels don't match")

    if projection_embedding.shape[0] != projection_labels.shape[0]:
        raise ValueError("The shape of the reference embedding and labels don't match")

    if sample is not None:
        n_samples = embedding.shape[0]
        sample_indices = np.random.choice(
            n_samples, size=min(sample, n_samples), replace=False
        )
        embedding = embedding[sample_indices]
        labels = labels[sample_indices]

        n_samples = projection_embedding.shape[0]
        sample_indices = np.random.choice(
            n_samples, size=min(sample, n_samples), replace=False
        )
        projection_embedding = projection_embedding[sample_indices]
        projection_labels = projection_labels[sample_indices]

    weights = projection_labels[:, None] == labels
    if not has_projection:
        np.fill_diagonal(weights, 0)

    mu = np.asarray(embedding.mean(axis=0)).ravel()

    numerator = np.sum(weights * ((projection_embedding - mu) @ (embedding - mu).T))
    denominator = np.sum((projection_embedding - mu) ** 2)

    return projection_embedding.shape[0] / np.sum(weights) * numerator / denominator
