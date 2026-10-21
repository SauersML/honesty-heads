"""Shared look for the graph-oracle figures: white, large type, no grid, direct labels, one palette."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, RED, GREEN, AMBER, PURPLE, TEAL = '#2F6DB5', '#D1495B', '#2E8B57', '#E09F3E', '#7B5EA7', '#2A9D8F'
SOFT = '#9DB9DD'  # light blue for a group drawn as background (not gray)
INK = '#222222'

def setup():
    plt.rcParams.update({
        'font.family': 'Helvetica Neue', 'font.size': 16, 'axes.titlesize': 20, 'axes.titleweight': 'bold',
        'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.edgecolor': INK,
        'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
        'axes.titlelocation': 'left', 'axes.titlepad': 16, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'lines.solid_capstyle': 'round'})

def clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_linewidth(1.0); ax.spines['bottom'].set_linewidth(1.0)
    ax.tick_params(length=4, width=1.0)

def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches='tight', pad_inches=0.3)
