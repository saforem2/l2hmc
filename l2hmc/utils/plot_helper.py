import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.pad'] = 3.5
plt.rcParams['xtick.major.size'] = 3.5
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['xtick.minor.pad'] = 3.4
plt.rcParams['xtick.minor.size'] = 2.0
plt.rcParams['xtick.minor.width'] = 0.6

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
markers = ['o', 's', 'x', 'v', 'h', '^', 'p', '<', 'd', '>', 'o']
linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
linestyles = 10 * ['']
#  linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
markersize = 3


def errorbar_plot(x_data, y_data, y_errors, out_file=None, **kwargs):
    """Create a single errorbar plot."""
    x = np.array(x_data)
    y = np.array(y_data)
    y_err = np.array(y_errors)
    if not (x.shape == y.shape == y_err.shape):
        import pdb
        pdb.set_trace()
        err_str = ("x, y, and y_errors all must have the same shape.\n"
                   f" x_data.shape: {x.shape}\n"
                   f" y_data.shape: {y.shape}\n"
                   f" y_errors.shape: {y_err.shape}")
        raise ValueError(err_str)
    #  x_label = axes_labels.get('x_label', '')
    #  y_label = axes_labels.get('y_label', '')
    #  if kwargs is not None:
    #  x_dim = kwargs.get('x_dim', 3)
    #  num_distributions = kwargs.get('num_distributions', 3)
    #  sigma = kwargs.get('sigma', 0.05)
    if kwargs is not None:
        markersize = kwargs.get('markersize', 3)
        x_label = kwargs.get('x_label', '')
        y_label = kwargs.get('y_label', '')
        title = kwargs.get('title', '')
        grid = kwargs.get('grid', False)
        reverse_x = kwargs.get('reverse_x', False)
        legend_labels = kwargs.get('legend_labels', np.full(x.shape[0], ''))
        plt_style = kwargs.get('plt_style', 'ggplot')
        try:
            plt.style.use(plt_style)
        except:
            plt.style.use('ggplot')

    fig, ax = plt.subplots()
    num_entries = x.shape[0]
    if num_entries > 1:
        fig, axes = plt.subplots(num_entries, sharex=True)
        for idx, ax in enumerate(axes):
            ax.errorbar(x[idx], y[idx], yerr=y_err[idx],
                        color=colors[idx], ls=linestyles[idx],
                        marker=markers[idx], markersize=markersize,
                        fillstyle='full', capsize=1.5, capthick=1.5,
                        alpha=0.75, label=legend_labels[idx])
            #  if idx == 0:
            #      ax.set_title(title, fontsize=16)
            #  ax.set_xlabel('', fontsize=10)
            if grid:
                ax.grid(True)
            if reverse_x:
                ax.set_xlim(ax.get_xlim()[::-1])
            ax.legend(loc='best', fontsize=10)

        axes[0].set_title(title, fontsize=16)
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        #  axes[-1].set_xlabel(x_label, fontsize=14)
        fig.tight_layout()

    else:
        ax.errorbar(x, y, yerr=y_err[idx],
                    color=colors[0], marker=markers[0],
                    ls=linestyles[0], fillstyle='full',
                    capsize=1.5, capthick=1.5, alpha=0.75,
                    label=legend_labels[0])
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best')
        if grid:
            ax.grid(True)
        if reverse_x:
            ax.set_xlim(ax.get_xlim()[::-1])

    if out_file is not None:
        print(f"Saving figure to: {out_file}")
        fig.tight_layout()
        plt.savefig(out_file, dpi=400, bbox_inches='tight')
    return fig, ax
