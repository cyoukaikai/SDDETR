from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
# def plot_bar(data_table: list, plot_name=None, labels=None,):
#
#     labels = ['G1', 'G2', 'G3', 'G4', 'G5']
#     men_means = [20, 34, 30, 35, 27]
#     women_means = [25, 32, 34, 20, 25]
#
#     x = np.arange(len(labels))  # the label locations
#     width = 0.7 / len(labels)  # the width of the bars
#
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - width / 2, men_means, width, label='Men')
#     rects2 = ax.bar(x + width / 2, women_means, width, label='Women')
#
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('Scores')
#     ax.set_title('Scores by group and gender')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()
#
#     ax.bar_label(rects1, padding=3)
#     ax.bar_label(rects2, padding=3)
#
#     fig.tight_layout()
#
#     if plot_name is None:
#         plt.show()
#     else:
#         plt.savefig(plot_name)
#         plt.close()


def plot_pie(ratios: list, plot_name=None, labels=None, explode=None, xlabel=None,
             additional_infor=None):
    """
    Pie chart, where the slices will be ordered and plotted counter-clockwise:
    # labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    # ratios = [15, 30, 45, 10]
    # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    # explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    @param additional_infor: additional information such as the total numbers
    @param explode:
    @param ratios:
    @param plot_name:
    @param labels:
    @return:
    """
    if explode is None:
        explode = tuple([0 for k in range(len(ratios))])

    fig1, ax1 = plt.subplots()

    if np.sum(ratios) != 100:
        total = np.sum(ratios)
        if labels is not None:
            if additional_infor is not None:
                labels = [f'{label}: {int(ratios[k])} \n {additional_infor[k]}'
                          for k, label in enumerate(labels)]
            else:
                labels = [f'{label}: {int(ratios[k])}' for k, label in enumerate(labels)]

        ratios = [x/total * 100 for x in ratios]

    ax1.pie(ratios, explode=explode, labels=labels, autopct='%1.1f%%',  #
            shadow=False, startangle=90)
    if xlabel is not None:
        # ax1.set_title(title)
        ax1.set_xlabel(xlabel)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    if plot_name is None:
        plt.show()
    else:
        plt.savefig(plot_name)
        plt.close()


def plot_matrix(matrix, plot_name='plot_matrix.jpg', vmin=None, vmax=None):
    # plt.figure()
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    # plt.imshow(matrix, vmin=vmin, vmax=vmax)
    plt.matshow(matrix, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if vmin is not None and vmax is not None:
        plt.clim(vmin, vmax)
    # plt.show()
    plt.savefig(plot_name)
    plt.close('all')


def plot_matrix_demo():
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(np.arange(100).reshape((10, 10)))

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def plot_distribution(xy, xy_labels, plot_file_name=None):
    fig = plt.figure()
    x, y = xy[0], xy[1]
    x = np.asarray(x, dtype=np.int32)  # numpy.ndarray
    y = np.asarray(y, dtype=np.float32)
    # plt.plot(x, y, label=self.dataset_name) #, label=label_name

    # count the occurrences of each point
    c = Counter(zip(x, y))
    # create a list of the sizes, here multiplied by 10 for scale
    area = [c[(xx, yy)] for xx, yy in zip(x, y)]
    plt.scatter(x, y, s=area)

    plt.xlabel(xy_labels[0], fontsize=14)
    plt.ylabel(xy_labels[1], fontsize=14)
    # plt.xlim(0, max() )
    # plt.ylim(0, 5 )
    # plt.title("image size statistics [ " + self.dataset_name + " ]")
    # plt.legend()
    if plot_file_name is not None:
        fig.savefig(plot_file_name)
        plt.close()
    else:
        plt.show()


def plot_histogram(data, plot_name,
                   title: str = None, xlabel: str = None, ylabel: str = None, fontsize: int = 14,
                   num_bin: int = None,
                   ):
    if num_bin is not None:
        plt.hist(data, bins=num_bin)  # arguments are passed to np.histogram
    else:
        plt.hist(data, bins='auto')  # arguments are passed to np.histogram
    # plt.title("Histogram with 'auto' bins")
    # Text(0.5, 1.0, "Histogram with 'auto' bins")

    if title is not None:
        plt.title(title, fontsize=fontsize)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)

    plt.savefig(plot_name)
    plt.close()


def plot_2d_feature(X, plot_name=None):
    fig = plt.figure()
    # ax = plt.gca()
    # plt.scatter(X[:, 0], X[:, 1], label='different objects', s=5, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], s=5)  # , cmap=plt.cm.Paired
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    if plot_name is not None:
        fig.savefig(plot_name)
        plt.close()
    else:
        plt.show()



# def plot_2d_point(X, y, plot_name=None):
#     if len(y[y == 0]) == 0 or len(y[y == 1]) == 0:
#         print(f'number of examples of label 0: {len(y[y == 0])}, '
#               f'number of examples of label 1: {len(y[y == 1])}')
#         return
#
#     fig = plt.figure()
#     ax = plt.gca()
#     X[:, 1] = 0
#     clf = svm.SVC(kernel='linear', C=1)
#     clf.fit(X, y)
#     for (intercept, coef) in zip(clf.intercept_, clf.coef_):
#         s = "y = {0:.3f}".format(intercept)
#         for (i, c) in enumerate(coef):
#             s += " + {0:.3f} * x{1}".format(c, i)
#         print(s)
#
#     # s = 30
#     # s=3,  s=10,  alpha=0.6, alpha=0.3, edgecolors='none' , label='different objects', ,  label='same object'
#     plt.scatter(X[y == 0, 0], X[y == 0, 1], label='different objects',
#                 )  # , cmap=plt.cm.Paired
#     plt.scatter(X[y == 1, 0], X[y == 1, 1], label='same object')  # , cmap=plt.cm.Paired
#     # plt.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap=plt.cm.Paired)
#     # plot the decision function
#     ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     plt.yticks([])
#
#     ax.legend()
#
#     # # create grid to evaluate model
#     # xx = np.linspace(xlim[0], xlim[1], 30)
#     # yy = np.linspace(ylim[0], ylim[1], 30)
#     xx = np.arange(0, 1, 0.05)
#     yy = np.linspace(ylim[0], ylim[1], 30)
#     YY, XX = np.meshgrid(yy, xx)
#     xy = np.vstack([XX.ravel(), YY.ravel()]).T
#     Z = clf.decision_function(xy).reshape(XX.shape)
#     plt.xlabel('similarity', fontsize=14)
#
#     # plt.ylabel('ambiguity', fontsize=14)
#     # plt.title(''.join(plot_name.split('.')[:-1]), fontsize=14)
#     # plot decision boundary and margins
#     ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#                linestyles=['--', '-', '--'])
#     # # # plot support vectors
#     # ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
#     #            linewidth=1, facecolors='none', edgecolors='k')
#     # plt.show()
#     if plot_name is None:
#         plt.savefig('eval_connection.jpg')
#     else:
#         plt.savefig(plot_name)
