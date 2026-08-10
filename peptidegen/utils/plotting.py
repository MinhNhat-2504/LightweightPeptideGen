import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

def setup_plot_style():
    """Configure modern, high-quality plot styling."""
    matplotlib.use('Agg') # Headless
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    })

def plot_histogram(data, title, xlabel, ylabel='Frequency', color='#4E79A7', bins=30, ax=None):
    if ax is None: fig, ax = plt.subplots()
    ax.hist(data, bins=bins, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def plot_bar_chart(labels, values, title, ylabel, color='#76B7B2', ax=None):
    if ax is None: fig, ax = plt.subplots()
    ax.bar(labels, values, color=color, alpha=0.85, edgecolor='black', linewidth=0.2)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    return ax
