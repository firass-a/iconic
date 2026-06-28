"""Generate Chapter 3 architecture figures (PNG). Run: python gen_ch3_figures.py"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.dirname(os.path.abspath(__file__))


def arch_diagram():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    boxes = [
        (0.3, 2.8, 'Weather\n(DSSAT CLI)', '#E8F5E9'),
        (0.3, 1.2, 'Soil profile\n(DSSAT)', '#E8F5E9'),
        (2.8, 2.0, 'DSSAT-CSM\nCERES-Maize', '#FFF3E0'),
        (5.5, 3.2, 'IoT sensor layer\n(5 virtual nodes)', '#E3F2FD'),
        (5.5, 1.0, 'Fault injector\n(noise, stuck,\nactuator η)', '#FFEBEE'),
        (8.2, 2.0, 'MORL agent\n(PC-PPO / CAPQL-LSTM)', '#F3E5F5'),
        (10.5, 2.0, 'Actuators\nN + irrigation', '#E8F5E9'),
    ]
    for x, y, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 1.8, 1.1, boxstyle='round,pad=0.05',
                                    facecolor=col, edgecolor='#333', linewidth=1.2))
        ax.text(x + 0.9, y + 0.55, txt, ha='center', va='center', fontsize=8)

    arrows = [
        ((2.1, 3.35), (2.8, 2.7)), ((2.1, 1.75), (2.8, 2.3)),
        ((4.6, 2.55), (5.5, 3.5)), ((4.6, 2.45), (5.5, 1.55)),
        ((7.3, 3.5), (8.2, 2.7)), ((7.3, 1.55), (8.2, 2.3)),
        ((10.0, 2.55), (10.5, 2.55)), ((11.3, 2.0), (4.6, 2.0)),
    ]
    for a, b in arrows:
        ax.annotate('', xy=b, xytext=a,
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.2))

    ax.text(6.0, 4.6, 'Proposed multi-objective resilient smart-farming framework',
            ha='center', fontsize=11, fontweight='bold')
    ax.text(6.0, 0.2, 'Closed loop: actions → DSSAT → crop state → (faulted) observations → agent',
            ha='center', fontsize=8, color='#555')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ch3_architecture.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Wrote ch3_architecture.png')


def interaction_loop():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')
    steps = ['Observe\nstate $s_t$', 'Select\naction $a_t$', 'Fault layer\n(optional)',
             'Execute\nN + water', 'DSSAT\nupdate', 'Reward\n$\\mathbf{R}_t$']
    xs = [0.2, 1.9, 3.6, 5.3, 7.0, 8.7]
    for i, (x, lbl) in enumerate(zip(xs, steps)):
        col = '#FFEBEE' if i == 2 else '#E3F2FD' if i in (0, 5) else '#FFF3E0' if i == 4 else '#F3E5F5'
        ax.add_patch(FancyBboxPatch((x, 1.0), 1.5, 1.2, boxstyle='round,pad=0.04',
                                    facecolor=col, edgecolor='#333'))
        ax.text(x + 0.75, 1.6, lbl, ha='center', va='center', fontsize=8)
        if i < len(xs) - 1:
            ax.annotate('', xy=(xs[i + 1] - 0.05, 1.6), xytext=(x + 1.55, 1.6),
                        arrowprops=dict(arrowstyle='->', color='#444', lw=1.5))
    ax.annotate('', xy=(0.2, 0.5), xytext=(8.7, 0.5),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(4.5, 0.35, 'next day $t+1$', ha='center', fontsize=8, color='#2E7D32')
    ax.set_title('Daily DSSAT–RL interaction loop', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ch3_interaction_loop.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Wrote ch3_interaction_loop.png')


if __name__ == '__main__':
    arch_diagram()
    interaction_loop()
