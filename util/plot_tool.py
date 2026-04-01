import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import packaging.version
import hist.intervals
import sys, os

sys.path.insert(0, 'util')
from common import fig_save_and_close, CheckDir
from copy import deepcopy

if packaging.version.parse(mpl.__version__) < packaging.version.parse("3.6"):
    MPL_STYLE = "seaborn-colorblind"
else:
    MPL_STYLE = "seaborn-v0_8-colorblind"


def plot_data_mc(Edges,
                 Contents,
                 config,
                 figure_path=None,
                 log_scale=False,
                 log_scale_x=False,
                 label="",
                 total_model_unc=None,
                 colors=None,
                 close_figure=False):
    nbin = 0
    region = []
    combined_contents = dict()
    region_edge = [0]
    for region_ in Edges:
        for hist_name in Contents[region_]:
            if hist_name not in combined_contents:
                combined_contents[hist_name] = deepcopy(Contents[region_][hist_name])
            else:
                combined_contents[hist_name]['Content'] = np.concatenate(
                    [combined_contents[hist_name]['Content'], Contents[region_][hist_name]['Content']], axis=0)
                combined_contents[hist_name]['Yield'] = combined_contents[hist_name]['Yield'] + \
                                                        Contents[region_][hist_name]['Yield']
        region.append(region_)
        nbin += len(Edges[region_]) - 1
        region_edge.append(region_edge[-1] + len(Edges[region_]) - 1)

    mc_histograms_yields = []
    mc_labels = []
    for hist_name in combined_contents:
        if combined_contents[hist_name]['Type'] == 'Data':
            data_histogram_yields = combined_contents[hist_name]['Content']
            data_histogram_interval = hist.intervals.poisson_interval(
                np.asarray(data_histogram_yields)
            )
            data_label = hist_name
        elif combined_contents[hist_name]['Type'] == 'Signal':
            signal_histogram_yields = combined_contents[hist_name]['Content']
            signal_label = hist_name
        else:
            mc_histograms_yields.append(combined_contents[hist_name]['Content'])
            mc_labels.append(hist_name)

    bin_edges = np.array([i for i in range(nbin + 1)])

    mpl.style.use(MPL_STYLE)
    fig = plt.figure(figsize=(6, 6), layout='constrained')
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # increase font sizes
    for item in (
            [ax1.yaxis.label, ax2.xaxis.label, ax2.yaxis.label]
            + ax1.get_yticklabels()
            + ax2.get_xticklabels()
            + ax2.get_yticklabels()
    ):
        item.set_fontsize("large")

    # minor ticks on all axes
    for axis in [ax1.xaxis, ax1.yaxis, ax2.xaxis, ax2.yaxis]:
        axis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    total_yield = np.zeros_like(mc_histograms_yields[0])
    bin_right_edges = bin_edges[1:]
    bin_left_edges = bin_edges[:-1]
    bin_width = bin_right_edges - bin_left_edges
    bin_centers = 0.5 * (bin_left_edges + bin_right_edges)
    # center data visually in bins if horizontal log scale is used
    bin_centers_data = (
        np.power(10, 0.5 * (np.log10(bin_left_edges * bin_right_edges)))
        if log_scale_x
        else bin_centers
    )
    mc_containers = []
    for mc_sample_yield, sample_label in zip(mc_histograms_yields, mc_labels):
        mc_container = ax1.bar(
            bin_centers,
            mc_sample_yield,
            width=bin_width,
            bottom=total_yield,
            color=colors[sample_label] if colors else None,
        )
        mc_containers.append(mc_container)

        # add a black line on top of each sample
        line_x = [y for y in bin_edges for _ in range(2)][1:-1]
        line_y = [y for y in (mc_sample_yield + total_yield) for _ in range(2)]
        ax1.plot(line_x, line_y, "-", color="black", linewidth=0.5)

        total_yield += mc_sample_yield

    if total_model_unc is None:
        total_model_unc = np.sqrt(total_yield)

    # add total MC uncertainty
    mc_unc_container = ax1.bar(
        bin_centers,
        2 * total_model_unc,
        width=bin_width,
        bottom=total_yield - total_model_unc,
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=3 * "/",
    )

    # plot data
    data_container = ax1.errorbar(
        bin_centers_data,
        data_histogram_yields,
        yerr=np.abs(data_histogram_yields - data_histogram_interval),
        fmt="o",
        color="k",
    )

    # plot signal
    multiplier = int(np.log10(np.max(total_yield) * 1.5 / np.max(signal_histogram_yields)))
    signal_histogram_yields *= 10 ** multiplier
    signal_line_x = [y for y in bin_edges for _ in range(2)][1:-1]
    signal_line_y = [y for y in (signal_histogram_yields) for _ in range(2)]
    signal_container = ax1.plot(signal_line_x, signal_line_y, "--", color="red", linewidth=2)
    signal_label = r"{}(x$10^{}$)".format(signal_label, multiplier)

    # ratio plot
    ax2.plot(
        [bin_left_edges[0], bin_right_edges[-1]],
        [1, 1],
        "--",
        color="black",
        linewidth=1,
    )  # reference line along y=1

    n_zero_pred = sum(total_yield == 0.0)  # number of bins with zero predicted yields
    if n_zero_pred > 0:
        print(
            f"predicted yield is zero in {n_zero_pred} bin(s), excluded from ratio plot"
        )
    nonzero_model_yield = total_yield != 0.0

    if np.any(total_yield < 0.0):
        raise ValueError(
            f"{label} total model yield has negative bin(s): {total_yield.tolist()}"
        )

    # add uncertainty band around y=1
    rel_mc_unc = total_model_unc / total_yield
    # do not show band in bins where total model yield is 0
    ax2.bar(
        bin_centers[nonzero_model_yield],
        2 * rel_mc_unc[nonzero_model_yield],
        width=bin_width[nonzero_model_yield],
        bottom=1.0 - rel_mc_unc[nonzero_model_yield],
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=3 * "/",
    )

    # data in ratio plot
    data_model_ratio = data_histogram_yields / total_yield
    data_model_ratio_unc = (
            np.abs(data_histogram_yields - data_histogram_interval) / total_yield
    )
    # mask data in bins where total model yield is 0
    ax2.errorbar(
        bin_centers_data[nonzero_model_yield],
        data_model_ratio[nonzero_model_yield],
        yerr=data_model_ratio_unc[:, nonzero_model_yield],
        fmt="o",
        color="k",
    )

    # get the highest single bin yield, from the sum of MC or data
    y_max = max(np.amax(total_yield), np.amax(data_histogram_yields))
    # lowest model yield in single bin (not considering empty bins)
    y_min = np.amin(total_yield[np.nonzero(total_yield)])

    # use log scale if it is requested, otherwise determine scale setting:
    # if yields vary over more than 2 orders of magnitude, set y-axis to log scale
    if log_scale or (log_scale is None and (y_max / y_min) > 100):
        # log vertical axis scale and limits
        ax1.set_yscale("log")
        ax1.set_ylim([y_min / 10, y_max * 10])
        # add "_log" to the figure name if figure should be saved
        # figure_path = utils._log_figure_path(figure_path)
    else:
        # do not use log scale
        ax1.set_ylim([0, y_max * 1.5])  # 50% headroom

        # log scale for horizontal axes
    if log_scale_x:
        ax1.set_xscale("log")
        ax2.set_xscale("log")

    # figure label (region name)
    at = mpl.offsetbox.AnchoredText(
        label,
        loc="upper left",
        frameon=False,
        prop={"fontsize": "large", "linespacing": 1.5},
    )
    ax1.add_artist(at)

    # Add region boundary
    for region_idx, boundary in enumerate(region_edge[1:-1]):
        ax1.axvline(x=boundary, color='black', linestyle='--')
        ax2.axvline(x=boundary, color='black', linestyle='--')
    for region_idx, boundary in enumerate(region_edge[1:]):
        ax2.text(0.5 * (region_edge[region_idx] + region_edge[region_idx + 1]), 1.2, region[region_idx],
                 horizontalalignment='center', fontsize=12, color='green', weight='bold')

    ax1.text(0.5 * (region_edge[0] + region_edge[1]), y_max * 1.2 if not log_scale else y_max * 2.5,
             r"Lumi = {} ".format(int(config.Lumi / 1000)) + r"fb$^{-1}$", fontsize=14, color='black',
             horizontalalignment='center', weight='bold')
    # MC contributions in inverse order, such that first legend entry corresponds to
    # the last (highest) contribution to the stack
    all_containers = mc_containers[::-1] + [mc_unc_container, data_container, signal_container[0]]
    all_labels = mc_labels[::-1] + ["Uncertainty", data_label, signal_label]
    ax1.legend(
        all_containers, all_labels, frameon=False, fontsize="large", loc="upper right"
    )
    ax1.set_xlim(bin_edges[0], bin_edges[-1])
    ax1.set_ylabel("events")
    ax1.set_xticklabels([])
    ax1.set_xticklabels([], minor=True)
    ax1.tick_params(axis="both", which="major", pad=8)  # tick label - axis padding
    ax1.tick_params(direction="in", top=True, right=True, which="both")

    ax2.set_xlim(bin_edges[0], bin_edges[-1])
    ax2.set_ylim([0.5, 1.5])
    ax2.set_xlabel(config.observable)
    ax2.set_ylabel("data / model")
    ax2.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax2.set_yticklabels([0.5, 0.75, 1.0, 1.25, ""])
    ax2.tick_params(axis="both", which="major", pad=8)
    ax2.tick_params(direction="in", top=True, right=True, which="both")
    fig_save_and_close(fig, path=figure_path, close_figure=close_figure)
    del ax1, ax2, fig
    # return fig


def plot_sig_bkg(X, Y, W, binning, variable_name, outdir, config):
    fig, ax = plt.subplots(figsize=(9, 6))
    X_bkg = X[Y == 0]
    X_sig = X[Y == 1]
    W_bkg = W[Y == 0]
    W_sig = W[Y == 1]

    h_bkg = ax.hist(X_bkg, binning, color='r', alpha=0.5, label='bkg', weights=W_bkg, density=True)
    h_sig = ax.hist(X_sig, binning, color='b', alpha=0.5, label='sig', weights=W_sig, density=True)

    ax.legend()
    plt.title("Distribution")
    plt.xlabel(variable_name)

    CheckDir(os.path.join(outdir, 'Distribution'))

    plt.savefig(os.path.join(outdir, 'Distribution', '{}.png'.format(variable_name)))
    plt.savefig(os.path.join(outdir, 'Distribution', '{}.pdf'.format(variable_name)))


def plot_train_test(x_train, y_train, w_train, x_test, y_test, w_test, binning, variable_name, outdir, config):
    fig, ax = plt.subplots(figsize=(9, 6))
    x_train_bkg = x_train[y_train == 0]
    x_train_sig = x_train[y_train == 1]
    x_test_bkg = x_test[y_test == 0]
    x_test_sig = x_test[y_test == 1]
    w_train_bkg = w_train[y_train == 0]
    w_train_sig = w_train[y_train == 1]
    w_test_bkg = w_test[y_test == 0]
    w_test_sig = w_test[y_test == 1]

    h_bkg = ax.hist(x_train_bkg, binning, color='r', alpha=0.5, label='bkg(train)', weights=w_train_bkg, density=True)
    h_sig = ax.hist(x_train_sig, binning, color='b', alpha=0.5, label='sig(train)', weights=w_train_sig, density=True)
    bkg, bins = np.histogram(x_test_bkg, binning, weights=w_test_bkg, density=True)
    sig, bins = np.histogram(x_test_sig, binning, weights=w_test_sig, density=True)
    ax.scatter(bins[:-1] + 0.5 * (bins[1:] - bins[:-1]), bkg, marker='o', c='red', s=40, alpha=1, label='bkg(test)')
    ax.scatter(bins[:-1] + 0.5 * (bins[1:] - bins[:-1]), sig, marker='o', c='blue', s=40, alpha=1, label='sig(test)')

    ax.legend()
    plt.title("Distribution")
    plt.xlabel(variable_name)

    CheckDir(os.path.join(outdir, 'Distribution'))

    plt.savefig(os.path.join(outdir, 'Distribution', '{}_train_v_test.png'.format(variable_name)))
    plt.savefig(os.path.join(outdir, 'Distribution', '{}_train_v_test.pdf'.format(variable_name)))

