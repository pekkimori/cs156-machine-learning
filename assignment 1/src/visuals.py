"""Pipeline diagram geometry and PNG rendering for the executed notebook.

The shared node positions and routes match the standalone TikZ diagram in
report.tex. Rendering uses saved labels and performs no model fitting.
"""
from pathlib import Path

PALETTE = {'data': '#FFB45D', 'model': '#A8C6F0', 'prediction': '#B7A9D8',
           'error': '#85D9A4', 'metric': '#F3E985', 'ink': '#263340'}
# Center x/y, width/height, semantic color, label. Shared by TikZ and PNG.
# Counts in these explanatory labels describe the saved assignment run. When
# using a different archive or split, update the labels from that run's audit;
# diagram geometry itself performs no data analysis or split computation.
NODES = {
    'raw': (5, 10.7, 4.4, .85, 'data', 'Personal XML + metadata'),
    'frame': (5, 9.1, 4.4, 1.0, 'data', 'Clean, joined dataframe\n294 rated completed titles'),
    'choices': (1.9, 7.4, 3.2, 1.05, 'model', 'Model conditions\n+ fixed settings'),
    'dev': (7, 7.4, 3.7, 1.05, 'data', 'Development data\n241 titles / 103 groups'),
    'test': (15.4, 7.4, 3.8, 1.05, 'data', 'Held-out test data\n53 titles / 26 groups'),
    'model': (1.9, 5.1, 2.8, .95, 'model', 'Untrained model\n+ preprocessing'),
    'train': (5.7, 5.1, 2.9, .95, 'data', 'Training folds\nX_train, y_train'),
    'val': (10.1, 5.1, 2.9, .95, 'data', 'Validation fold\nX_val, y_val'),
    'fit': (3.8, 3.2, 3.2, .95, 'model', 'Fitted preprocessing\n+ trained model'),
    'pred': (7.6, 3.2, 2.7, .95, 'prediction', 'Validation\npredictions'),
    'err': (10.7, 3.2, 2.6, .95, 'error', 'Validation errors'),
    'cvmetrics': (7.6, 1.45, 3.2, .95, 'metric', 'Mean CV metrics\nMAE / favorite AP'),
    'selected': (7.6, -.35, 3.7, .95, 'model', 'Select using CV;\nrefit on development'),
    'testpred': (15.4, 3.2, 3.5, .95, 'prediction', 'Held-out predictions'),
    'testerr': (15.4, 1.45, 3.5, .95, 'error', 'Held-out errors'),
    'final': (15.4, -.35, 3.5, .95, 'metric', 'Final metrics\n+ diagnostic plots'),
}
# Edges: source.anchor, target.anchor, intermediate points, optional label/x/y.
EDGES = [
    ('raw.south', 'frame.north', [], ('Parse, validate, join', 8.1, 9.95)),
    ('frame.south', 'dev.north', [(5, 8.3), (7, 8.3)], None),
    ('frame.south', 'test.north', [(5, 8.3), (15.4, 8.3)], ('Group-disjoint split', 11.5, 8.65)),
    ('choices.south', 'model.north', [], None),
    ('dev.south', 'train.north', [(7, 6.65), (5.7, 6.65)], None),
    ('dev.south', 'val.north', [(7, 6.65), (10.1, 6.65)], None),
    ('model.south', 'fit.north', [], None),
    ('train.south', 'fit.north', [], ('Fit on train', 5.7, 4.02)),
    ('fit.east', 'pred.west', [], None),
    ('val.south', 'pred.north', [], ('X_val', 8.1, 4.2)),
    ('val.south', 'err.north', [], ('y_val', 11.1, 4.2)),
    ('pred.east', 'err.west', [], None),
    ('err.south', 'cvmetrics.east', [(10.7, 1.45)], None),
    ('cvmetrics.south', 'selected.north', [], None),
    ('selected.east', 'testpred.west', [(12.55, -.35), (12.55, 3.2)], None),
    ('test.south', 'testpred.north', [], ('X_test', 15.95, 5.1)),
    ('test.east', 'testerr.east', [(18, 7.4), (18, 1.45)], ('y_test', 18.5, 4.5)),
    ('testpred.south', 'testerr.north', [], None),
    ('testerr.south', 'final.north', [], None),
]


def anchor(name):
    """Resolve 'node.side' to a diagram-space boundary coordinate.

    Node geometry stores a center and width/height. Adding or subtracting half
    that dimension places an arrow endpoint on the named north/south/east/west
    boundary. The same named anchors are referenced directly by TikZ.
    """
    node, side = name.split('.')
    x, y, w, h, *_ = NODES[node]
    return {'north': (x, y+h/2), 'south': (x, y-h/2),
            'west': (x-w/2, y), 'east': (x+w/2, y)}[side]


def draw_pipeline(path):
    """Save the color-coded flow diagram as a PNG at the supplied local path.

    NODES supplies box geometry and semantic colors; EDGES supplies endpoints,
    optional routing corners, and edge labels. Coordinates are diagram units,
    independent of the image's pixel resolution. The gray background encloses
    development CV, keeping held-out evaluation visibly outside that loop.

    Arrows are drawn behind boxes so their endpoints meet box boundaries.
    Labels are drawn last. Fixed limits and padded export bounds keep all
    outer arrows and annotations visible. This function changes no data.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.path import Path as MplPath
    fig, ax = plt.subplots(figsize=(12.8, 8))
    fig.subplots_adjust(left=.015, right=.985, top=.985, bottom=.015)
    ax.add_patch(FancyBboxPatch((0, .65), 12.1, 5.6,
        boxstyle='round,pad=.12,rounding_size=.3', facecolor='#F0F1F3',
        edgecolor='#A6ACB3', linewidth=1.1, zorder=0))
    ax.text(2.6, 1.47, 'Five-fold grouped\ncross-validation\n(development only)',
            ha='center', va='center', fontsize=10, color=PALETTE['ink'])
    for start, end, via, label in EDGES:
        # MOVETO starts a route, and subsequent LINETO commands follow each
        # explicit corner. The arrowhead appears only at the final endpoint.
        points = [anchor(start), *via, anchor(end)]
        path_data = MplPath(points, [MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path_data, arrowstyle='-|>',
            mutation_scale=15, linewidth=1.7, color=PALETTE['ink'], zorder=1))
        if label:
            text, x, y = label
            ax.text(x, y, text, ha='center', va='center', fontsize=9,
                    color=PALETTE['ink'], zorder=3,
                    bbox={'facecolor':'white','edgecolor':'none','pad':1.2,'alpha':.92})
    for x, y, w, h, color, label in NODES.values():
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle='round,pad=.02,rounding_size=.15', facecolor=PALETTE[color],
            edgecolor='#52606D', linewidth=1.1, zorder=2))
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                color=PALETTE['ink'], zorder=3)
    ax.set(xlim=(-.4, 19.2), ylim=(-1.1, 11.5))
    ax.set_aspect('equal'); ax.axis('off')
    fig.savefig(Path(path), dpi=190, bbox_inches='tight', pad_inches=.16)
    plt.close(fig)
