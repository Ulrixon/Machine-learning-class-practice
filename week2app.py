#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:35:04 2020

@author: kian
"""

# This file is for producing the figure of the comparison, the data comes from the week2.py
def plot_multi(data, cols=None, spacing=.3, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = getattr(getattr(plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0], fontsize=16)
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n], fontsize=16)

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax

data = pd.concat([pd.DataFrame(L, columns=['Empirical Loss']),pd.DataFrame(CVm,columns=['CV'])],axis=1)
fig3 = plt.figure(figsize=(5,4), dpi=200)
plot_multi(data, lw=3, fontsize=16)
plt.savefig(file_path+'/fig3.eps', format='eps', bbox_inches = 'tight')
    