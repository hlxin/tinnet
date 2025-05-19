#!/usr/bin/env python
# This script is adapted from Xie's and Ulissi's scripts.


import numpy as np
import matplotlib.pyplot as plt

from ase import io
from pylab import *

from ..prediction.image2property import Image2property

class Plot_SHAP:
    def __init__(self):
        self.model = Image2property()

    def plot_shap(self,
                  reference_image=None,
                  reference_site_idx=None,
                  reference_name='Reference',
                  target_image=None,
                  target_site_idx=None,
                  target_name='Target',
                  plot_name='shap'):
        
        rcParams['ps.useafm'] = True
        plt.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
        rcParams['pdf.fonttype'] = 42
        rcParams['errorbar.capsize'] = 4.0
        mpl.rcParams['ytick.major.width'] = 0.5
        mpl.rcParams['ytick.minor.width'] = 0.5
        mpl.rcParams['xtick.major.width'] = 0.5
        mpl.rcParams['xtick.minor.width'] = 0.5
        matplotlib.rc('xtick.major', size=4)
        matplotlib.rc('xtick.minor', size=2)
        matplotlib.rc('ytick.major', size=4)
        matplotlib.rc('ytick.minor', size=2)
        matplotlib.rc('lines', linewidth = 0.5)
        matplotlib.rc('lines', markeredgewidth=0.5)
        matplotlib.rc('font', size=7)
        plt.rcParams['axes.linewidth'] = 0.5
        
        fig, ax = plt.subplots()
        fig.set_size_inches(3.375*2.0, 3.375)
        
        shap = self.model.calculate_shap(reference_image,
                                         reference_site_idx,
                                         target_image,
                                         target_site_idx)
        
        shap = np.average(shap, axis=1)
        
        dx5 = shap[2]
        dx4 = shap[3]
        dx3 = shap[4]
        dx2 = shap[5]
        dx1 = shap[6]
        
        x5 = shap[0]
        x4 = x5 + dx5
        x3 = x4 + dx4
        x2 = x3 + dx3
        x1 = x2 + dx2
        
        c5 = (shap[2] < 0) * 'red' + (shap[2] >= 0) * 'blue'
        c4 = (shap[3] < 0) * 'red' + (shap[3] >= 0) * 'blue'
        c3 = (shap[4] < 0) * 'red' + (shap[4] >= 0) * 'blue'
        c2 = (shap[5] < 0) * 'red' + (shap[5] >= 0) * 'blue'
        c1 = (shap[6] < 0) * 'red' + (shap[6] >= 0) * 'blue'
        
        ax.arrow(x=x5, y=5, dx=dx5, dy=0, color=c5, width=1.0/3.0,
                 head_width=1.0/3.0, head_length=0.15*np.abs(dx5),
                 length_includes_head=True)
        ax.arrow(x=x4, y=4, dx=dx4, dy=0, color=c4, width=1.0/3.0,
                 head_width=1.0/3.0, head_length=0.15*np.abs(dx4),
                 length_includes_head=True)
        ax.arrow(x=x3, y=3, dx=dx3, dy=0, color=c3, width=1.0/3.0,
                 head_width=1.0/3.0, head_length=0.15*np.abs(dx3),
                 length_includes_head=True)
        ax.arrow(x=x2, y=2, dx=dx2, dy=0, color=c2, width=1.0/3.0,
                 head_width=1.0/3.0, head_length=0.15*np.abs(dx2),
                 length_includes_head=True)
        ax.arrow(x=x1, y=1, dx=dx1, dy=0, color=c1, width=1.0/3.0,
                 head_width=1.0/3.0, head_length=0.15*np.abs(dx1),
                 length_includes_head=True)
        
        ax.set_ylim([0.5, 5.5])
        
        labels = [r'$(\alpha, \xi)$',
                  r'$d_{ij}$',
                  r'$\zeta$',
                  r'$(\lambda, r_{dj})$',
                  r'$(\beta, \Delta\chi)$']
        
        plt.yticks([5,4,3,2,1], labels)
        
        ax.annotate('Resonance', xy=(-0.12, 5.0),
                    xycoords=('axes fraction', 'data'),
                    ha='center', va='center')
        ax.annotate('Ligand', xy=(-0.12, 2.0),
                    xycoords=('axes fraction', 'data'),
                    ha='center', va='center')
        ax.annotate('Charge\ntransfer', xy=(-0.12, 1.0),
                    xycoords=('axes fraction', 'data'),
                    ha='center', va='center')
        
        ax.annotate('Strain',
                    xy=(-0.06, 3.5), xycoords=('axes fraction', 'data'),
                    xytext=(-0.12, 3.5), textcoords=('axes fraction', 'data'),
                    arrowprops=dict(arrowstyle='-[, widthB=3.0, lengthB=1.0'),
                    ha='center', va='center')
        
        sym_5 = (shap[2] < 0) * '-' + (shap[2] >= 0) * '+'
        sym_4 = (shap[3] < 0) * '-' + (shap[3] >= 0) * '+'
        sym_3 = (shap[4] < 0) * '-' + (shap[4] >= 0) * '+'
        sym_2 = (shap[5] < 0) * '-' + (shap[5] >= 0) * '+'
        sym_1 = (shap[6] < 0) * '-' + (shap[6] >= 0) * '+'
        
        ax.annotate(sym_5 + '{:.4f}'.format(round(np.abs(shap[2]), 4)),
                    xy=(-0.12, 4.70), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', color=c5)
        ax.annotate(sym_4 + '{:.4f}'.format(round(np.abs(shap[3]), 4)),
                    xy=(-0.12, 3.80), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', color=c4)
        ax.annotate(sym_3 + '{:.4f}'.format(round(np.abs(shap[4]), 4)),
                    xy=(-0.12, 3.20), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', color=c3)
        ax.annotate(sym_2 + '{:.4f}'.format(round(np.abs(shap[5]), 4)),
                    xy=(-0.12, 1.70), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', color=c2)
        ax.annotate(sym_1 + '{:.4f}'.format(round(np.abs(shap[6]), 4)),
                    xy=(-0.12, 0.60), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', color=c1)
        
        ax.set_xlabel(r'$d\rm{-band\ center}$ (eV)')
        ax.spines[['left', 'right', 'top']].set_visible(False)
        
        ax.tick_params('y', length=0, width=0, which='major')
        
        ax.plot([x5, x5],[7, 0.5],'--', color='gray', linewidth=1)
        
        ax.plot([x4, x4], [5 + 1.0 / 3.0, 4 - 1.0 / 3.0],'--', color='gray',
                linewidth=1)
        ax.plot([x3, x3], [4 + 1.0 / 3.0, 3 - 1.0 / 3.0],'--', color='gray',
                linewidth=1)
        ax.plot([x2, x2], [3 + 1.0 / 3.0, 2 - 1.0 / 3.0],'--', color='gray',
                linewidth=1)
        ax.plot([x1, x1], [2 + 1.0 / 3.0, 1 - 1.0 / 3.0],'--', color='gray',
                linewidth=1)
        
        ax.plot([x1 + dx1, x1 + dx1],[7, 0.5],'--', color='orange',
                linewidth=1)
        
        ax.annotate(reference_name + '\n{:.4f}'.format(round(x5, 4)),
                    xy=(x5, 1.15), xycoords=('data', 'axes fraction'),
                    ha='center', va='center', color='gray')
        ax.annotate(target_name + '\n{:.4f}'.format(round(x1 + dx1, 4)),
                    xy=(x1 + dx1, 1.05), xycoords=('data', 'axes fraction'),
                    ha='center', va='center', color='orange')
        
        fig.tight_layout()
        
        fig.savefig(plot_name + '.pdf', bbox_inches='tight')
