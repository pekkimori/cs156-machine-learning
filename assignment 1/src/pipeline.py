"""Leakage-aware, offline experiments. Run from the repository root.

All representations are fit inside each training fold. Fixed settings and seed
are defined before inspecting held-out performance. Results are written to disk.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
# Fixed semantic vocabulary, chosen without reference to ratings. These are
# broad genre/theme tags in an aggregated, non-curated taxonomy, not MAL genres.
TAGS = {'action', 'adventure', 'comedy', 'drama', 'fantasy', 'horror', 'mystery',
        'romance', 'sci-fi', 'science fiction', 'slice of life', 'sports',
        'supernatural', 'suspense', 'thriller', 'psychological', 'mecha', 'music',
        'historical', 'military', 'school', 'school life', 'shounen', 'shoujo',
        'seinen', 'josei', 'isekai', 'ecchi', 'harem', 'parody', 'super power',
        'martial arts', 'vampire', 'vampires', 'space', 'survival', 'time travel',
        'detective', 'detectives', 'workplace', 'iyashikei', 'gore', 'samurai'}
FEATURES = ['tags', 'studios', 'type', 'year', 'log_episodes']


def split_labels(value):
    """Split pipe-delimited categories, preserving multiword labels as tokens.

    CountVectorizer calls this tokenizer for tags and studios. Empty segments
    are discarded, and a studio such as 'white fox co., ltd.' stays one token.
    """
    return [s for s in value.split('|') if s]


def load_data():
    """Join bundled archive rows to metadata and prepare raw feature columns.

    Returns:
        A DataFrame containing every archive entry, its audit fields, relation
        group, and the columns in FEATURES. The one-to-one merge rejects
        duplicate IDs. Ratings and viewing status remain available for sample
        selection but are excluded from FEATURES.

    Only deterministic transformations occur here: a fixed tag whitelist,
    category formatting, and log episode counts. Learned vocabularies, medians,
    and scaling statistics are fitted later within each training partition.
    """
    archive = pd.DataFrame(json.loads((ROOT / 'data/archive.json').read_text()))
    meta = pd.DataFrame(json.loads((ROOT / 'data/metadata_snapshot.json').read_text()))
    df = archive.merge(meta, on='anime_id', how='left', validate='one_to_one')
    df['tags'] = df['tags'].map(lambda t: '|'.join(sorted(TAGS.intersection(t))))
    df['studios'] = df['studios'].map(lambda t: '|'.join(t) if t else 'Unknown')
    df['type'] = df['type'].fillna(df['xml_type']).replace('UNKNOWN', 'Unknown')
    # Raw export counts are used consistently; 0 denotes an unknown count.
    df['log_episodes'] = np.log1p(df['xml_episodes'].replace(0, np.nan))
    df['year'] = pd.to_numeric(df['year'], errors='coerce').replace(0, np.nan)
    return df


def preprocessing(columns=FEATURES):
    """Construct an unfitted encoder for a requested feature-group subset.

    Args:
        columns: Raw feature groups to retain. Ablations pass smaller lists.

    Returns:
        A ColumnTransformer that creates binary tag/studio indicators, one-hot
        format columns, and imputed/scaled numeric columns. Dense output works
        with every model in this small comparison.

    Calling this function learns nothing. Pipeline.fit learns vocabularies and
    numerical statistics from its training rows. Unknown validation categories
    contribute no learned indicator, and numeric imputation uses training data.
    Each text block requires at least one token in its training partition.
    """
    blocks = []
    for name in ['tags', 'studios']:
        if name in columns:
            blocks.append((name, CountVectorizer(tokenizer=split_labels,
                token_pattern=None, lowercase=False, binary=True), name))
    if 'type' in columns:
        blocks.append(('type', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['type']))
    nums = [c for c in ['year', 'log_episodes'] if c in columns]
    if nums:
        # Missingness indicators preserve whether a numeric value was absent.
        # The main sample has complete numeric data, so no indicators are added
        # for those columns in this dataset's ordinary training fits.
        blocks.append(('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
            ('scale', StandardScaler())]), nums))
    return ColumnTransformer(blocks, sparse_threshold=0)


def regressors():
    """Return fresh estimators for the five fixed score-prediction conditions.

    The constants learn the training mean or median. Ridge regularizes feature
    weights, the tree limits partition depth and leaf size, and the network
    uses two nonlinear hidden layers. Every stochastic estimator uses SEED.

    TransformedTargetRegressor fits target scaling on the labels supplied to
    its fit call and reverses that scaling during predict. Thus validation
    labels cannot influence the network's target normalization. Numeric L2
    settings belong to each estimator's own objective and target scale.
    """
    return {
        'Mean baseline': DummyRegressor(strategy='mean'),
        'Median baseline': DummyRegressor(strategy='median'),
        'Ridge regression': Ridge(alpha=10),
        'Decision tree': DecisionTreeRegressor(max_depth=4, min_samples_leaf=8, random_state=SEED),
        'Neural network': TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu',
                solver='lbfgs', alpha=10, max_iter=3000, random_state=SEED),
            transformer=StandardScaler())}


def score_regression(y, p):
    """Calculate complementary score and favorite-ranking metrics.

    Args:
        y: Observed scores, in the same positional order as p.
        p: Continuous predictions; clipping enforces the valid range [1, 10].

    Returns:
        MAE and RMSE; equal-weight MAE across observed exact score values;
        MAE for the fixed 1-6 analysis band; fraction within one rating point;
        AP using scores to rank 10/10 favorites; and Spearman rank correlation.
        Lower-band MAE is NaN when that band has no observations. Spearman is
        NaN for constant predictions, for which rank correlation is undefined.

    Macro_score_MAE gives each observed score equal weight. Ordinary MAE gives
    each title equal weight. Predictions are continuous and are never rounded.
    """
    y, p = np.asarray(y), np.clip(p, 1, 10)
    by_score = [np.abs(p[y == s] - s).mean() for s in np.unique(y)]
    lower = y <= 6
    return {'MAE': mean_absolute_error(y, p),
            'RMSE': np.sqrt(mean_squared_error(y, p)),
            'Macro_score_MAE': np.mean(by_score),
            'Lower_MAE': mean_absolute_error(y[lower], p[lower]) if lower.any() else np.nan,
            'Within_one': np.mean(np.abs(p-y) <= 1),
            'Favorite_AP': average_precision_score(y == 10, p),
            'Spearman': float(spearmanr(y, p).statistic) if np.std(p) > 1e-10 else np.nan}


def make_pipe(model, columns=FEATURES):
    """Create an independent, unfitted preprocessing-and-model pipeline.

    Cloning copies estimator settings and discards fitted state. Every fold
    therefore gets its own transformer and estimator. Dummy estimators use the
    common pipeline structure and ignore the resulting explanatory features.
    """
    return Pipeline([('features', preprocessing(columns)), ('model', clone(model))])


def fit_cv(X, y, splits, model, columns=FEATURES):
    """Fit and score one model on a reusable set of positional CV splits.

    Args:
        X, y: Development features and labels with matching row order.
        splits: Train/validation integer-position pairs. Callers use disjoint
            relation groups and a partition that validates each row once.
        model: Estimator configuration to clone for every fold.
        columns: Feature groups to encode, including any ablation restriction.

    Returns:
        (fold_metrics, out_of_fold_predictions, warning_messages). Predictions
        occupy the original development-row positions, enabling aligned checks.
        Fold metrics include training MAE to expose train/validation gaps.

    Fitting the complete pipeline inside the loop prevents vocabulary, numeric
    preprocessing, and estimator state from leaking into validation. Warnings
    are retained for the audit so failed convergence cannot be hidden by a
    successful return. The caller averages fold errors with equal fold weight.
    """
    values, fold_rows, captured = np.zeros(len(y)), [], []
    for fold, (tr, va) in enumerate(splits):
        pipe = make_pipe(model, columns)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            pipe.fit(X.iloc[tr], y.iloc[tr])
        captured.extend(str(w.message) for w in ws)
        pred = np.clip(pipe.predict(X.iloc[va]), 1, 10)
        # va contains positions in the original development sample, so this
        # assignment restores alignment after folds are evaluated separately.
        values[va] = pred
        fold_rows.append({'fold': fold, 'n_validation': len(va),
            'train_MAE': mean_absolute_error(y.iloc[tr], np.clip(pipe.predict(X.iloc[tr]), 1, 10)),
            **score_regression(y.iloc[va], pred)})
    return pd.DataFrame(fold_rows), values, captured


def run_experiments():
    """Execute the fixed offline protocol and save its audit trail.

    Returns:
        A dictionary of the primary sample, audit summary, score comparison,
        favorite comparison, ablations, held-out predictions, learning curve,
        band errors, and development-fit ridge coefficients.

    Side effects:
        Creates or overwrites named CSV/JSON files under results/ and figures
        under figures/. Raw bundled inputs are read without modification. No
        live API or repository operation is performed.

    Protocol:
        1. Reserve whole relation groups as test data with a fixed seed.
        2. Compare fixed conditions using grouped development-only CV.
        3. Select by validation metrics, then evaluate the fixed test sample.
        4. Report conditional bootstrap uncertainty and development diagnostics.

    The selected model is never replaced after seeing held-out performance.
    Learning curves, ablations, and the population sensitivity remain confined
    to development groups. The test-band analysis describes the final errors.
    """
    (ROOT / 'results').mkdir(exist_ok=True)
    df = load_data()
    # Completed works define the primary estimand. Dropped and partial ratings
    # remain in the audit and a development-only sensitivity analysis.
    data = df[(df.score > 0) & (df.status == 'Completed')].copy().reset_index(drop=True)
    X, y, groups = data[FEATURES], data.score, data.relation_group
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(X, y, groups))
    assert not set(groups.iloc[dev]).intersection(groups.iloc[test])
    # dev/test are positions in the primary sample; CV positions are local to
    # development. Reset indices so positional arrays and pandas labels align.
    Xd, yd, gd = X.iloc[dev].reset_index(drop=True), y.iloc[dev].reset_index(drop=True), groups.iloc[dev].reset_index(drop=True)
    Xt, yt, gt = X.iloc[test].reset_index(drop=True), y.iloc[test].reset_index(drop=True), groups.iloc[test].reset_index(drop=True)
    cv = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xd, yd, gd))
    for tr, va in cv:
        assert not set(gd.iloc[tr]).intersection(gd.iloc[va])
    data['split'] = 'development'
    data.loc[test, 'split'] = 'test'
    fold_of = {}
    # Map local validation positions back to primary-sample positions. The
    # exported manifest lets a reader audit every title's actual assignment.
    for fold, (_, va) in enumerate(cv):
        fold_of.update({int(dev[j]): fold for j in va})
    data['cv_fold'] = [fold_of.get(i, -1) for i in range(len(data))]
    data[['anime_id', 'title', 'score', 'relation_group', 'split', 'cv_fold']].to_csv(ROOT / 'results/split_manifest.csv', index=False)
    cv_rows, oof, all_warnings = [], {}, []
    # Reuse the identical fold boundaries for all conditions. Differences then
    # reflect model behavior on the same validation groups.
    for name, model in regressors().items():
        folds, preds, ws = fit_cv(Xd, yd, cv, model)
        all_warnings.extend(ws)
        folds.insert(0, 'model', name)
        cv_rows.append(folds)
        oof[name] = preds
    fold_results = pd.concat(cv_rows, ignore_index=True)
    cv_table = fold_results.groupby('model', sort=False).agg(
        CV_MAE=('MAE', 'mean'), CV_SD=('MAE', 'std'), Train_MAE=('train_MAE', 'mean'))
    # Selection occurs before any held-out predictions or errors are computed.
    selected = cv_table.CV_MAE.idxmin()
    # Keep the overall winner and best learned condition separately: a constant
    # baseline is allowed to win, while diagnostics still examine a learned fit.
    learned = cv_table.loc[['Ridge regression', 'Decision tree', 'Neural network'], 'CV_MAE'].idxmin()
    test_preds, test_rows, fitted = {}, [], {}
    for name, model in regressors().items():
        pipe = make_pipe(model).fit(Xd, yd)
        p = np.clip(pipe.predict(Xt), 1, 10)
        fitted[name], test_preds[name] = pipe, p
        test_rows.append({'model': name, **score_regression(yt, p)})
    table = cv_table.join(pd.DataFrame(test_rows).set_index('model'))
    table.to_csv(ROOT / 'results/model_comparison.csv')
    fold_results.to_csv(ROOT / 'results/cv_folds.csv', index=False)
    predictions = data.iloc[test][['anime_id', 'title', 'score', 'relation_group']].reset_index(drop=True)
    for name, p in test_preds.items():
        predictions[name] = p
    predictions.to_csv(ROOT / 'results/test_predictions.csv', index=False)
    # A different scientific question: rank favorites (10) against everything
    # else. AP is evaluated using probabilities, with no test-set threshold fit.
    favorite_models = {'Prior baseline': DummyClassifier(strategy='prior'),
        'Favorite logistic': LogisticRegression(C=.1, class_weight='balanced', max_iter=3000, random_state=SEED)}
    favorite_folds, favorite_pipes = [], {}
    # Derive the binary target separately within each training/validation slice.
    # predict_proba[:, 1] refers to class 1: the exact score-10 favorite label.
    # Weighted logistic outputs support ranking; calibration is not assumed.
    for name, model in favorite_models.items():
        for fold, (tr, va) in enumerate(cv):
            pipe = make_pipe(model).fit(Xd.iloc[tr], (yd.iloc[tr] == 10).astype(int))
            prob = pipe.predict_proba(Xd.iloc[va])[:, 1]
            favorite_folds.append({'model': name, 'fold': fold,
                'AP': average_precision_score(yd.iloc[va] == 10, prob)})
        favorite_pipes[name] = make_pipe(model).fit(Xd, (yd == 10).astype(int))
    favorite_cv = pd.DataFrame(favorite_folds).groupby('model', sort=False).AP.agg(['mean', 'std'])
    favorite_selected = favorite_cv['mean'].idxmax()
    # Favorite selection is recorded before any favorite test probabilities
    # are scored. The fixed 0.5 decision threshold receives no test tuning.
    favorite_rows = []
    for name, pipe in favorite_pipes.items():
        prob = pipe.predict_proba(Xt)[:, 1]
        favorite_rows.append({'model': name, 'CV_AP': favorite_cv.loc[name, 'mean'],
            'CV_SD': favorite_cv.loc[name, 'std'], 'Test_AP': average_precision_score(yt == 10, prob),
            'ROC_AUC': roc_auc_score(yt == 10, prob),
            'Balanced_accuracy': balanced_accuracy_score(yt == 10, prob >= .5),
            'Brier': brier_score_loss(yt == 10, prob)})
        predictions[name] = prob
    favorite_table = pd.DataFrame(favorite_rows).set_index('model')
    favorite_table.to_csv(ROOT / 'results/favorite_comparison.csv')
    predictions.to_csv(ROOT / 'results/test_predictions.csv', index=False)
    # Cluster bootstrap preserves dependence among related test entries. This
    # estimates uncertainty conditional on this particular fitted split/model.
    rng = np.random.default_rng(SEED)
    unique = gt.unique()
    draws = []
    diffs = np.abs(test_preds[learned] - yt.to_numpy()) - np.abs(test_preds['Median baseline'] - yt.to_numpy())
    # One paired error difference per held-out title: negative favors the learned
    # model. Reuse the SAME resampled positions for both models through diffs.
    for _ in range(2000):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        # A sampled group contributes all its titles. Drawing it twice repeats
        # its entire block. Bootstrap replicates can have different title counts
        # because relation groups have unequal sizes. No model is retrained.
        ix = np.concatenate([np.flatnonzero(gt.to_numpy() == g) for g in chosen])
        draws.append(diffs[ix].mean())
    # Development-only learning curve using group subsets, with the same fixed
    # learned-model settings and untouched validation groups.
    learning = []
    for fold, (tr, va) in enumerate(cv):
        train_groups = np.unique(gd.iloc[tr])
        rng = np.random.default_rng(SEED + fold)
        rng.shuffle(train_groups)
        # Shuffle once per fold, then take nested prefixes of the group list.
        # Fractions apply to groups; record actual title counts for the plot.
        for frac in [.3, .6, 1.]:
            use = train_groups[:max(2, int(np.ceil(frac * len(train_groups))))]
            subset = np.array([j for j in tr if gd.iloc[j] in use])
            pipe = make_pipe(regressors()[learned]).fit(Xd.iloc[subset], yd.iloc[subset])
            learning.append({'fraction': frac, 'fold': fold, 'n_train': len(subset),
                'Train_MAE': mean_absolute_error(yd.iloc[subset], np.clip(pipe.predict(Xd.iloc[subset]), 1, 10)),
                'Validation_MAE': mean_absolute_error(yd.iloc[va], np.clip(pipe.predict(Xd.iloc[va]), 1, 10))})
    learning = pd.DataFrame(learning)
    learning.to_csv(ROOT / 'results/learning_curve.csv', index=False)
    # Fixed ridge ablations isolate whether broad tags and studios help; each
    # variant is refitted inside the same development folds, never on test data.
    ablations = []
    for label, cols in [('All features', FEATURES), ('No tags', [c for c in FEATURES if c != 'tags']),
                        ('No studios', [c for c in FEATURES if c != 'studios']),
                        ('Type + numeric', ['type', 'year', 'log_episodes'])]:
        folds, _, ws = fit_cv(Xd, yd, cv, Ridge(alpha=10), cols)
        all_warnings.extend(ws)
        ablations.append({'condition': label, 'CV_MAE': folds.MAE.mean(), 'CV_SD': folds.MAE.std()})
    ablations = pd.DataFrame(ablations)
    ablations.to_csv(ROOT / 'results/ablations.csv', index=False)
    # Add partial/dropped ratings only to development groups; keep every test
    # component excluded. This is a population sensitivity, not a new test run.
    extended = df[(df.score > 0) & ~df.relation_group.isin(set(gt))].reset_index(drop=True)
    # The enlarged population gets a new CV partition. Its scores describe that
    # population and cannot isolate the added rows' effect in a paired estimate.
    ex_cv = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(extended[FEATURES], extended.score, extended.relation_group))
    sensitivity = []
    for name in ['Median baseline', 'Ridge regression']:
        fs, _, _ = fit_cv(extended[FEATURES], extended.score, ex_cv, regressors()[name])
        sensitivity.append({'population': 'All rated in development groups', 'model': name,
                            'n': len(extended), 'CV_MAE': fs.MAE.mean()})
    pd.DataFrame(sensitivity).to_csv(ROOT / 'results/population_sensitivity.csv', index=False)
    bands = []
    for label, mask in [('10: favorites', yt == 10), ('9: liked a lot', yt == 9),
                         ('7-8: less impact', yt.isin([7, 8])), ('1-6: lower ratings', yt <= 6)]:
        for name, p in test_preds.items():
            bands.append({'band': label, 'model': name, 'n': int(mask.sum()),
                'MAE': mean_absolute_error(yt[mask], p[mask]) if mask.any() else np.nan})
    bands = pd.DataFrame(bands)
    bands.to_csv(ROOT / 'results/error_by_band.csv', index=False)
    # Coefficients describe conditional associations in development data only.
    ridge = fitted['Ridge regression']
    # ColumnTransformer names follow transformed-column order, matching coef_.
    # This fit saw all development labels and none of the test labels. Binary
    # coefficients use indicator units; numeric coefficients use scaled units.
    coefficients = pd.DataFrame({'feature': ridge['features'].get_feature_names_out(),
                                 'coefficient': ridge['model'].coef_})
    coefficients.sort_values('coefficient').to_csv(ROOT / 'results/ridge_coefficients.csv', index=False)
    audit = {'archive_entries': len(df), 'rated_entries': int((df.score > 0).sum()),
        'unrated_entries': int((df.score == 0).sum()), 'primary_entries': len(data),
        'excluded_rated_incomplete': int(((df.score > 0) & (df.status != 'Completed')).sum()),
        'matched_entries': int(df.matched.sum()), 'score_counts_all_rated': df[df.score > 0].score.value_counts().sort_index().to_dict(),
        'score_counts_completed': data.score.value_counts().sort_index().to_dict(),
        'primary_mean': data.score.mean(), 'primary_median': data.score.median(),
        'dev_n': len(dev), 'test_n': len(test), 'dev_groups': gd.nunique(), 'test_groups': gt.nunique(),
        'primary_groups': groups.nunique(), 'largest_primary_group': int(groups.value_counts().max()),
        'test_favorites': int((yt == 10).sum()), 'test_lower': int((yt <= 6).sum()),
        'missing_year': int(data.year.isna().sum()), 'missing_episodes': int(data.log_episodes.isna().sum()),
        'empty_tags': int((data.tags == '').sum()), 'unknown_studios': int((data.studios == 'Unknown').sum()),
        'encoded_features_dev': len(ridge['features'].get_feature_names_out()),
        'selected_by_cv': selected, 'best_learned_by_cv': learned,
        'favorite_selected_by_cv': favorite_selected,
        'learned_minus_median_test_MAE': float(diffs.mean()),
        'bootstrap_95_interval': np.quantile(draws, [.025, .975]).tolist(),
        'warnings': sorted(set(all_warnings)),
        'cv_fold_sizes': [len(v) for _, v in cv],
        'episode_count_disagreements': int((data.xml_episodes != data.episodes).sum())}
    (ROOT / 'results/audit.json').write_text(json.dumps(audit, indent=2, default=int) + '\n')
    make_figures(data, table, favorite_table, learning, predictions, bands, learned)
    return {'data': data, 'audit': audit, 'comparison': table, 'favorites': favorite_table,
            'ablations': ablations, 'predictions': predictions, 'learning': learning,
            'bands': bands, 'coefficients': coefficients}


def make_figures(data, table, favorite, learning, predictions, bands, learned):
    """Render the five assignment figures from completed experiment tables.

    Args:
        data: Primary completed/rated sample for the rating distribution.
        table, favorite: Score and favorite metric tables indexed by model.
        learning: Development learning-curve rows for each fraction and fold.
        predictions: Held-out labels and prediction columns in matching order.
        bands: Per-model test-band errors and sample counts.
        learned: Name of the learned condition chosen using development MAE.

    Saves PNGs under figures/ and closes every Matplotlib figure. No fitting or
    metric-based model selection occurs. Horizontal jitter is seeded and used
    only for drawing overlapping score points; stored predictions stay intact.
    Tight bounding boxes plus explicit margins preserve edge labels and marks.
    """
    folder = ROOT / 'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 130, 'savefig.dpi': 190})
    fig, ax = plt.subplots(figsize=(8.4, 3.2), layout='constrained')
    counts = data.score.value_counts().reindex(range(1, 11), fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=['#b27832']*6 + ['#526879']*2 + ['#167e8d', '#244a63'])
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set(xlabel='Personal score (0 = unrated, excluded)', ylabel='Completed titles', xticks=range(1, 11),
           title='Ratings are concentrated near the top of the scale')
    ax.set_ylim(0, counts.max()*1.2)
    ax.set_xlim(.25, 10.75)
    fig.savefig(folder / 'rating_distribution.png', bbox_inches='tight', pad_inches=.18); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained', gridspec_kw={'width_ratios':[1.6, 1]})
    x = np.arange(len(table))
    axes[0].barh(x+.17, table.CV_MAE, height=.32, label='5-fold development CV', color='#9aa7b2')
    axes[0].barh(x-.17, table.MAE, height=.32, label='Held-out groups', color='#167e8d')
    axes[0].set(yticks=x, yticklabels=table.index, xlabel='Mean absolute error (lower is better)', title='Score prediction')
    axes[0].margins(y=.12)
    axes[0].set_xlim(0, table[['CV_MAE', 'MAE']].to_numpy().max()*1.18)
    axes[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.19),
                   ncol=2, frameon=False)
    x = np.arange(len(favorite))
    axes[1].bar(x-.18, favorite.CV_AP, width=.35, label='CV', color='#9aa7b2')
    axes[1].bar(x+.18, favorite.Test_AP, width=.35, label='Held out', color='#167e8d')
    axes[1].set(xticks=x, xticklabels=['Prior', 'Logistic'], ylim=(0, 1), ylabel='Average precision (higher is better)', title='Finding 10/10 favorites')
    axes[1].set_xlim(-.65, 1.65)
    axes[1].legend(fontsize=8)
    fig.savefig(folder / 'model_comparison.png', bbox_inches='tight', pad_inches=.18); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    rng = np.random.default_rng(SEED)
    for ax, name in zip(axes, ['Median baseline', learned]):
        ax.scatter(predictions.score + rng.uniform(-.10, .10, len(predictions)), predictions[name],
                   s=26, alpha=.65, color='#167e8d')
        ax.plot([1,10],[1,10], '--', color='#526879', linewidth=1)
        ax.set(xlim=(.6,10.4), ylim=(.6,10.4), xticks=[1,3,5,7,9,10],
               xlabel='Actual personal rating (horizontal jitter)', ylabel='Predicted score', title=name)
    fig.savefig(folder / 'heldout_predictions.png', bbox_inches='tight', pad_inches=.18); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    avg = learning.groupby('fraction').mean(numeric_only=True)
    axes[0].plot(avg.n_train, avg.Train_MAE, 'o-', color='#526879', label='Training')
    axes[0].plot(avg.n_train, avg.Validation_MAE, 'o-', color='#167e8d', label='Validation')
    axes[0].set(xlabel='Mean training titles per fold', ylabel='MAE', title=f'Learning curve: {learned}')
    axes[0].margins(x=.12, y=.18)
    axes[0].legend(fontsize=8)
    subset = bands[bands.model.isin(['Median baseline', learned])].pivot(index='band', columns='model', values='MAE')
    subset = subset.reindex(['10: favorites', '9: liked a lot', '7-8: less impact', '1-6: lower ratings'])
    x = np.arange(len(subset))
    for j, name in enumerate(['Median baseline', learned]):
        axes[1].bar(x + (j-.5)*.35, subset[name], width=.35, label=name, color=['#9aa7b2','#167e8d'][j])
    ns = bands[bands.model == learned].set_index('band').n
    axes[1].set(xticks=x, xticklabels=[f'{s.split(":")[0]}\n(n={ns[s]})' for s in subset.index],
                ylabel='MAE', title='Held-out errors by preference band')
    axes[1].set_xlim(-.65, 3.65)
    axes[1].set_ylim(0, subset.to_numpy().max()*1.35)
    axes[1].legend(fontsize=8)
    fig.savefig(folder / 'learning_and_tail_errors.png', bbox_inches='tight', pad_inches=.18); plt.close(fig)
    # The diagram shares its layout and semantics with the standalone source.
    try:
        from .visuals import draw_pipeline
    except ImportError:
        from visuals import draw_pipeline
    draw_pipeline(folder / 'pipeline.png')


if __name__ == '__main__':
    result = run_experiments()
    print(json.dumps(result['audit'], indent=2, default=int))
    print(result['comparison'].round(3).to_string())
    print(result['favorites'].round(3).to_string())
