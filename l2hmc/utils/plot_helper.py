import os
import numpy as np
import matplotlib.pyplot as plt

#  try:
#      plt.style.use('/Users/saforem2/.config/matplotlib/'
#                     + 'stylelib/ggplot_sam.mplstyle')
#
#  except:
#      plt.style.use('ggplot')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.pad'] = 3.5
plt.rcParams['xtick.major.size'] = 3.5
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['xtick.minor.pad'] = 3.4
plt.rcParams['xtick.minor.size'] = 2.0
plt.rcParams['xtick.minor.width'] = 0.6

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
MARKERS = ['o', 's', 'x', 'v', 'h', '^', 'p', '<', 'd', '>', 'o']
LINESTYLES = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
#  linestyles = 10 * ['']
#  linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--']
MARKERSIZE = 3


# pylint: disable=too-many-arguments
def plot_broken_xaxis(xdata, 
                      ydata, 
                      xlabel, 
                      ylabel, 
                      output_file=None,
                      xlim1=(-2, 100),
                      xlim2=(398, 500)):
    """Create plot with a broken x-axis."""
    # pylint: disable=invalid-name
    fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
    # plot the same data on both axes
    for idx in range(ydata.shape[1]):
        _ = ax.plot(xdata, ydata[:, idx], marker='', ls='-',
                    alpha=0.7, lw=1.5, label=f'sample {idx}')

    _ = ax.plot(xdata, ydata.mean(axis=1), marker='', ls='-',
                color='k', lw=2., label='average')

    for idx in range(ydata.shape[1]):
        _ = ax2.plot(xdata, ydata[:, idx], marker='', ls='-',
                     alpha=0.7, lw=1.5, label=f'sample {idx}')

    _ = ax2.plot(xdata, ydata.mean(axis=1), marker='', ls='-',
                 color='k', lw=2., label='average')

    # zoom-in / limit the view to different portions of the data
    _ = ax.set_xlim(xlim1)
    _ = ax2.set_xlim(xlim2)

    # hide the spines between ax and ax2
    _ = ax.spines['right'].set_visible(False)
    _ = ax2.spines['left'].set_visible(False)
    _ = ax.yaxis.tick_left()
    _ = ax.tick_params(labelright=False)
    _ = ax2.yaxis.tick_right()

    # pylint: disable=invalid-name
    d = 0.015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    _ = ax.plot((1-d, 1+d), (-d, +d), **kwargs)
    _ = ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    _ = kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    _ = ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    _ = ax2.plot((-d, +d), (-d, +d), **kwargs)

    _ = ax.set_ylabel(ylabel, fontsize=14)
    _ = ax.set_xlabel(xlabel, fontsize=14)
    _ = ax.xaxis.set_label_coords(1.1, -0.065)
    _ = ax2.legend(loc='best', fontsize=10)

    _ = plt.show()  # pylint: disable=F841

    if output_file is not None:
        fig.savefig(output_file, dpi=400, bbox_inches='tight')

    return fig, ax, ax2


# pylint: disable=too-many-statements,too-many-locals
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
        capsize = kwargs.get('capsize', 1.5)
        capthick = kwargs.get('capthick', 1.5)
        fillstyle = kwargs.get('fillstyle', 'full')
        alpha = kwargs.get('alpha', 0.8)
        markersize = kwargs.get('markersize', 3)
        x_label = kwargs.get('x_label', '')
        y_label = kwargs.get('y_label', '')
        title = kwargs.get('title', '')
        grid = kwargs.get('grid', False)
        reverse_x = kwargs.get('reverse_x', False)
        legend_labels = kwargs.get('legend_labels', np.full(x.shape[0], ''))

    num_entries = x.shape[0]
    if num_entries > 1:
        fig, axes = plt.subplots(num_entries, sharex=True)
        for idx, ax in enumerate(axes):
            ax.errorbar(x[idx], y[idx], yerr=y_err[idx],
                        color=COLORS[idx], ls=LINESTYLES[idx],
                        marker=MARKERS[idx], markersize=markersize,
                        fillstyle=fillstyle, alpha=alpha,
                        capsize=capsize, capthick=capthick,
                        label=legend_labels[idx])
            #  if idx == 0:
            #      ax.set_title(title, fontsize=16)
            #  ax.set_xlabel('', fontsize=10)
            if grid:
                ax.grid(True)
            if reverse_x:
                ax.set_xlim(ax.get_xlim()[::-1])
            ax.legend(loc='best', fontsize=10, markerscale=1.5)

        axes[0].set_title(title, fontsize=16)
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        axes[-1].set_xlabel(x_label, fontsize=14)
        fig.tight_layout()

    else:
        fig, ax = plt.subplots()
        ax.errorbar(x, y, yerr=y_err[idx],
                    color=COLORS[0], marker=MARKERS[0],
                    ls=LINESTYLES[0], fillstyle='full',
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

    if num_entries > 1:
        return fig, axes
    else: 
        return fig, ax


# pylint: disable=too-many-statements,too-many-locals
def annealing_schedule_plot(**kwargs):
    """Plot annealing schedule."""
    train_steps = kwargs.get('num_training_steps')
    temp_init = kwargs.get('temp_init')
    annealing_factor = kwargs.get('annealing_factor')
    annealing_steps = kwargs.get('annealing_steps')
    annealing_steps_init = kwargs.get('_annealing_steps_init')
    tunneling_steps = kwargs.get('tunneling_rate_steps')
    tunneling_steps_init = kwargs.get('_tunneling_rate_steps_init')
    figs_dir = kwargs.get('figs_dir')
    steps_arr = kwargs.get('steps_arr')
    temp_arr = kwargs.get('temp_arr')
    #  num_steps = max(steps_arr)
    max_steps_arr = max(steps_arr)
    num_steps = max(train_steps, max_steps_arr)

    #steps = np.arange(num_steps)
    temps = []
    steps = []
    temp = temp_init
    for step in range(num_steps):
        if step % annealing_steps_init == 0:
            tt = temp * annealing_factor
            if tt > 1:
                temp = tt
        if (step+1) % tunneling_steps_init == 0:
            steps.append(step+1)
            temps.append(temp)

    fig, ax = plt.subplots()
    _ = ax.axhline(y=1., color='C6', ls='-', lw=2., label='T=1')
    _ = ax.plot(steps, temps, ls='--', label='Fixed schedule', lw=2)
    _ = ax.plot(steps_arr, temp_arr,
                label='Dynamic schedule', lw=2, alpha=0.75)
    _ = ax.set_xlabel('Training step')
    _ = ax.set_ylabel('Temperature')
    _ = ax.legend(loc='best')
    print(f"Saving figure to: {figs_dir}annealing_schedule.pdf")

    plt.savefig(figs_dir + 'annealing_schedule.pdf',
                dpi=400, bbox_inches='tight')
    return fig, ax
