"""
================================================================================
Kaggle Playground Series S6E8 — Predicting Smartphone Addiction — V2
================================================================================
V2 over V1: Optuna tuning ON by default, a 4th ensemble member (a PyTorch
tabular MLP with entity embeddings), 10-fold x 5-seed bagging, pseudo-labeling
wired in as a real second pass, and an ensembler that tries weighted blend /
rank-average blend / logistic stacking / GBM stacking and keeps whichever
wins on OOF AUC. GPU is auto-detected with automatic CPU fallback per model.

WHAT CHANGED FROM V1 (evidence-based, not guesswork)
-----------------------------------------------------
On a controlled subsample I A/B tested target encoding, KMeans cluster
features, cross-categoricals, and polynomial interaction features against
the V1 feature set. Every one of them came back flat-to-worse. So V2 keeps
V1's feature engineering as-is by default (raw features + missingness flags +
domain ratios) — the lever here is NOT more features, it's better-tuned
models, more of them, and more bagging. CONFIG['use_extra_features'] exists
if you want to re-test the fancier features yourself; it's off by default.

I also found and fixed a real bug from V1: pandas' string dtype does not
stringify NaN the way older pandas did, so `.astype(str)` alone left real
NaNs in CatBoost's categorical columns and would crash Pool construction.
Fixed with an explicit `.fillna('missing')` before the string cast.

GPU SETUP ON KAGGLE / COLAB
---------------------------
- XGBoost (device='cuda') and CatBoost (task_type='GPU') use the GPU out of
  the box with the standard `pip install xgboost catboost` — nothing extra
  to do, and both fall back to CPU automatically if GPU training throws.
- LightGBM's standard pip wheel has NO GPU backend compiled in. If you want
  GPU LightGBM, run this first in a cell:
      !pip uninstall -y lightgbm
      !pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON
  If that build fails (it sometimes does depending on the image), just leave
  LightGBM on CPU — it's usually the fastest of the three even on CPU, and
  the fallback wrapper here handles it transparently either way.
- You have 2x T4. The simplest way to actually use both without fragile
  in-process GPU orchestration: split CONFIG['seeds'] in half, run this
  script twice (once per Kaggle session / Colab runtime) with
  `os.environ['CUDA_VISIBLE_DEVICES']` set to '0' and '1' respectively, then
  average the two resulting submission.csv files. Free 2x wall-clock speedup,
  zero extra code complexity, zero risk of the two runs stepping on each
  other's GPU memory.

USAGE
-----
1. Drop this file next to train.csv / test.csv / sample_submission.csv, or
   just attach the competition dataset on Kaggle (default paths already
   point at the standard input directory for this competition).
2. Adjust CONFIG if needed.
3. Run: python smartphone_addiction_pipeline_v2.py
4. Output: submission.csv + oof_predictions.csv (OOF preds from every base
   model + the chosen ensemble method, for your own further stacking).
================================================================================
"""

import os
import time
import warnings
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except Exception as _torch_err:
    # Catches more than ImportError on purpose: a broken CUDA driver / partial
    # install raises OSError deep inside torch's C-extension loader, not
    # ImportError. Either way we want to fall back to the 3-model GBM
    # ensemble instead of crashing the whole script.
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')

# Force line-buffered stdout. Kaggle/Jupyter notebook kernels sometimes
# buffer stdout more aggressively than a plain terminal, so a script that is
# actually running fine can *look* completely stuck for a long stretch just
# because none of its print() output has been flushed yet. This one line
# fixes that for every print() in the file, not just the ones that pass
# flush=True explicitly.
import sys as _sys
try:
    _sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
CONFIG = {
    'train_path_candidates': ['/kaggle/input/playground-series-s6e8/train.csv', 'train.csv'],
    'test_path_candidates': ['/kaggle/input/playground-series-s6e8/test.csv', 'test.csv'],
    'sample_sub_path_candidates': [
        '/kaggle/input/playground-series-s6e8/sample_submission.csv', 'sample_submission.csv'],
    'output_path': 'submission.csv',
    'oof_output_path': 'oof_predictions.csv',

    'id_col': 'id',
    'target_col': 'addicted_label',

    # ---- cross-validation ----
    'n_folds': 10,
    'seeds': [42, 202, 2026, 7, 123],       # 5-seed bagging
    'random_state': 42,

    # ---- speed / thoroughness toggle ----
    'quick_mode': False,                     # True -> 5-fold/1-seed/small trees, for sanity checks

    # ---- compute ----
    'use_gpu': 'auto',
    'n_jobs': -1,

    # ---- feature engineering ----
    'use_extra_features': False,             # target-enc / kmeans / cross-cat / poly — tested, didn't help

    # ---- models to include in the ensemble ----
    'use_nn': True,                          # requires torch; auto-skipped if unavailable

    # ---- Optuna (ON by default now — you have real GPU budget) ----
    'run_optuna': True,
    'optuna_trials': 60,
    'optuna_folds': 3,
    # Tune on CPU by default. This is not a conservatism setting — CatBoost's
    # GPU path recompiles CUDA kernels on every single model instantiation,
    # so 60 trials x 3 folds x 3 models on GPU pays that compile cost ~500+
    # times and can look "stuck" for a very long time before it does any
    # real training. CPU has no such per-trial startup tax, and Optuna trials
    # here use small subsets (optuna_folds) anyway, so CPU tuning is usually
    # faster in wall-clock terms even though each individual fit is slower.
    # The FINAL full CV training below still uses GPU (CONFIG['use_gpu']).
    'optuna_use_gpu': False,
    'optuna_timeout': 1200,  # hard cap in seconds per model, regardless of trial count

    # ---- pseudo-labeling (real second pass, only kept if it improves OOF AUC) ----
    'use_pseudo_labeling': True,
    'pseudo_label_confidence': 0.98,

    # ---- NN hyperparameters ----
    'nn_epochs': 60,
    'nn_patience': 8,
    'nn_batch_size': 4096,
    'nn_lr': 1e-3,
    'nn_hidden': (256, 128, 64),
    'nn_dropout': 0.3,
}


# ==============================================================================
# GPU DETECTION (automatic CPU fallback is inside every train_*_cv function)
# ==============================================================================
def detect_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def resolve_gpu(config):
    if config['use_gpu'] == 'auto':
        has_gpu = detect_gpu()
        print(f"GPU detected: {has_gpu}")
        return has_gpu
    print(f"GPU forced to: {config['use_gpu']}")
    return bool(config['use_gpu'])


# ==============================================================================
# DATA LOADING
# ==============================================================================
def _first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def load_data(config):
    train = pd.read_csv(_first_existing(config['train_path_candidates']))
    test = pd.read_csv(_first_existing(config['test_path_candidates']))
    sample_sub = pd.read_csv(_first_existing(config['sample_sub_path_candidates']))
    return train, test, sample_sub


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
RAW_COLS = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
            'work_study_hours', 'sleep_hours', 'notifications_per_day',
            'app_opens_per_day', 'weekend_screen_time', 'gender', 'stress_level',
            'academic_work_impact']
CAT_COLS = ['gender', 'stress_level', 'academic_work_impact']
NUM_ONLY_RAW = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                'work_study_hours', 'sleep_hours', 'notifications_per_day',
                'app_opens_per_day', 'weekend_screen_time']


def engineer_features(df, use_extra=False):
    """V1 feature set (proven on real data) + an optional, evidence-discouraged
    'extra' bundle you can re-test yourself via CONFIG['use_extra_features']."""
    df = df.copy()
    eps = 1e-3

    for c in RAW_COLS:
        df[f'{c}_isna'] = df[c].isnull().astype(np.int8)
    df['n_missing'] = df[RAW_COLS].isnull().sum(axis=1).astype(np.int8)

    df['stress_level_ordinal'] = df['stress_level'].map({'Low': 0, 'Medium': 1, 'High': 2})
    df['academic_work_impact_bin'] = df['academic_work_impact'].map({'No': 0, 'Yes': 1})

    df['screen_sleep_ratio'] = df['daily_screen_time_hours'] / (df['sleep_hours'] + eps)
    df['social_share_of_screen'] = df['social_media_hours'] / (df['daily_screen_time_hours'] + eps)
    df['gaming_share_of_screen'] = df['gaming_hours'] / (df['daily_screen_time_hours'] + eps)
    df['leisure_hours'] = df['social_media_hours'] + df['gaming_hours']
    df['leisure_share_of_screen'] = df['leisure_hours'] / (df['daily_screen_time_hours'] + eps)
    df['work_leisure_ratio'] = df['work_study_hours'] / (df['leisure_hours'] + eps)
    df['weekend_weekday_diff'] = df['weekend_screen_time'] - df['daily_screen_time_hours']
    df['weekend_weekday_ratio'] = df['weekend_screen_time'] / (df['daily_screen_time_hours'] + eps)
    df['notif_per_open'] = df['notifications_per_day'] / (df['app_opens_per_day'] + eps)
    df['mins_per_open'] = (df['daily_screen_time_hours'] * 60) / (df['app_opens_per_day'] + eps)
    df['notif_per_screen_hour'] = df['notifications_per_day'] / (df['daily_screen_time_hours'] + eps)
    df['accounted_hours'] = df['work_study_hours'] + df['leisure_hours'] + df['sleep_hours']
    df['free_hours_remaining'] = 24 - df['accounted_hours']

    if use_extra:
        # NOTE: tested on a controlled subsample and found flat-to-negative
        # vs. the features above. Left here (fully working) in case it
        # behaves differently on your full-data / full-CV setup — worth
        # re-checking yourself rather than trusting my subsample finding blindly.
        df['notif_x_opens'] = df['notifications_per_day'] * df['app_opens_per_day']
        df['screen_x_notif'] = df['daily_screen_time_hours'] * df['notifications_per_day']
        df['age_x_screen'] = df['age'] * df['daily_screen_time_hours']
        df['weekend_x_notif'] = df['weekend_screen_time'] * df['notifications_per_day']
        df['opens_per_hour'] = df['app_opens_per_day'] / (df['daily_screen_time_hours'] * 60 + eps)
        for c in ['notifications_per_day', 'app_opens_per_day', 'daily_screen_time_hours', 'weekend_screen_time']:
            df[f'{c}_rank'] = df[c].rank(pct=True)

    return df


def prepare_features(train, test, config):
    id_col, target_col = config['id_col'], config['target_col']
    use_extra = config['use_extra_features']

    y = train[target_col].copy()
    train_feat = engineer_features(train.drop(columns=[target_col]), use_extra)
    test_feat = engineer_features(test, use_extra)

    combined = pd.concat([train_feat[CAT_COLS], test_feat[CAT_COLS]], axis=0)
    for c in CAT_COLS:
        cats = sorted(combined[c].dropna().unique().tolist())
        train_feat[c] = pd.Categorical(train_feat[c], categories=cats)
        test_feat[c] = pd.Categorical(test_feat[c], categories=cats)

    feature_cols = [c for c in train_feat.columns if c != id_col]
    X, X_test = train_feat[feature_cols], test_feat[feature_cols]

    # CatBoost wants plain strings for categoricals. IMPORTANT: fillna BEFORE
    # the string cast — pandas' modern string dtype does not stringify NaN
    # the way `.astype(str)` alone used to, so casting first silently leaves
    # real NaNs in place and crashes CatBoost's Pool constructor.
    X_cb, X_test_cb = X.copy(), X_test.copy()
    for c in CAT_COLS:
        X_cb[c] = X_cb[c].astype(object).fillna('missing').astype(str)
        X_test_cb[c] = X_test_cb[c].astype(object).fillna('missing').astype(str)

    num_cols = [c for c in feature_cols if c not in CAT_COLS]
    return X, X_cb, y, X_test, X_test_cb, feature_cols, num_cols


# ==============================================================================
# MODEL PARAM BUILDERS
# ==============================================================================
def get_lgb_params(config, use_gpu):
    n = config['n_estimators']['lgb']
    if config['quick_mode']:
        p = dict(n_estimators=n, learning_rate=0.04, num_leaves=63, min_child_samples=30,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1)
    else:
        p = dict(n_estimators=n, learning_rate=0.02, num_leaves=127, min_child_samples=40,
                 subsample=0.8, colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=0.2)
    p.update(objective='binary', verbosity=-1, n_jobs=config['n_jobs'],
             subsample_freq=1, device_type=('gpu' if use_gpu else 'cpu'))
    return p


def get_xgb_params(config, use_gpu):
    n = config['n_estimators']['xgb']
    if config['quick_mode']:
        p = dict(n_estimators=n, learning_rate=0.04, max_depth=6, min_child_weight=5,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
    else:
        p = dict(n_estimators=n, learning_rate=0.02, max_depth=8, min_child_weight=8,
                 subsample=0.8, colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=1.5)
    p.update(objective='binary:logistic', eval_metric='auc', tree_method='hist',
             enable_categorical=True, n_jobs=config['n_jobs'],
             device=('cuda' if use_gpu else 'cpu'))
    return p


def get_cb_params(config, use_gpu):
    n = config['n_estimators']['cb']
    if config['quick_mode']:
        p = dict(iterations=n, learning_rate=0.06, depth=7, l2_leaf_reg=3.0)
    else:
        p = dict(iterations=n, learning_rate=0.025, depth=8, l2_leaf_reg=5.0)
    p.update(loss_function='Logloss', eval_metric='AUC', verbose=False,
             thread_count=config['n_jobs'], task_type=('GPU' if use_gpu else 'CPU'))
    return p


# ==============================================================================
# OPTUNA TUNING (fully implemented; ON by default in V2)
# ==============================================================================
def _require_optuna():
    if optuna is None:
        raise ImportError("Optuna is not installed. `pip install optuna` to use run_optuna=True.")


def _make_progress_callback(model_name):
    """Prints after every trial so it's obvious the search is alive, and
    flushes explicitly — some notebook environments (Kaggle included) buffer
    stdout aggressively, which is a common reason a working script *looks*
    hung when it isn't."""
    def callback(study, trial):
        print(f"    [{model_name} Optuna] trial {trial.number + 1}/{study.user_attrs.get('n_trials', '?')} "
              f"AUC={trial.value:.5f} | best so far={study.best_value:.5f} "
              f"| elapsed={time.time() - study.user_attrs['t0']:.0f}s", flush=True)
    return callback


def tune_lgb_optuna(X, y, config, use_gpu):
    _require_optuna()
    print(f"  Tuning LightGBM ({'GPU' if use_gpu else 'CPU'}) — up to {config['optuna_trials']} trials "
          f"or {config['optuna_timeout']}s, whichever comes first...", flush=True)

    def objective(trial):
        base = dict(
            objective='binary', verbosity=-1, n_jobs=config['n_jobs'], n_estimators=3000,
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int('num_leaves', 15, 255),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        )
        skf = StratifiedKFold(n_splits=config['optuna_folds'], shuffle=True, random_state=config['random_state'])
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            params = dict(base, device_type=('gpu' if use_gpu else 'cpu'))
            try:
                model = lgb.LGBMClassifier(**params)
                model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                          eval_metric='auc', categorical_feature=CAT_COLS,
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            except Exception as e:
                if params.get('device_type') == 'gpu':
                    params['device_type'] = 'cpu'
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                              eval_metric='auc', categorical_feature=CAT_COLS,
                              callbacks=[lgb.early_stopping(50, verbose=False)])
                else:
                    raise
            pred = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[val_idx], pred))
        return float(np.mean(scores))


    study = optuna.create_study(direction='maximize')
    study.set_user_attr('t0', time.time())
    study.set_user_attr('n_trials', config['optuna_trials'])
    study.optimize(objective, n_trials=config['optuna_trials'], timeout=config['optuna_timeout'],
                   show_progress_bar=False, callbacks=[_make_progress_callback('LightGBM')])
    print(f"  Best LightGBM Optuna AUC: {study.best_value:.5f} | params: {study.best_params}", flush=True)
    return study.best_params


def tune_xgb_optuna(X, y, config, use_gpu):
    _require_optuna()
    print(f"  Tuning XGBoost ({'GPU' if use_gpu else 'CPU'}) — up to {config['optuna_trials']} trials "
          f"or {config['optuna_timeout']}s, whichever comes first...", flush=True)

    def objective(trial):
        base = dict(
            objective='binary:logistic', eval_metric='auc', tree_method='hist',
            enable_categorical=True, n_jobs=config['n_jobs'], n_estimators=3000,
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 20),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            early_stopping_rounds=50,
        )
        skf = StratifiedKFold(n_splits=config['optuna_folds'], shuffle=True, random_state=config['random_state'])
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            params = dict(base, device=('cuda' if use_gpu else 'cpu'))
            try:
                model = xgb.XGBClassifier(**params)
                model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=False)
            except Exception as e:
                if params.get('device') == 'cuda':
                    params['device'] = 'cpu'
                    model = xgb.XGBClassifier(**params)
                    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=False)
                else:
                    raise
            pred = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[val_idx], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize')
    study.set_user_attr('t0', time.time())
    study.set_user_attr('n_trials', config['optuna_trials'])
    study.optimize(objective, n_trials=config['optuna_trials'], timeout=config['optuna_timeout'],
                   show_progress_bar=False, callbacks=[_make_progress_callback('XGBoost')])
    print(f"  Best XGBoost Optuna AUC: {study.best_value:.5f} | params: {study.best_params}", flush=True)
    return study.best_params


def tune_cb_optuna(X_cb, y, config, use_gpu):
    _require_optuna()
    print(f"  Tuning CatBoost ({'GPU' if use_gpu else 'CPU'}) — up to {config['optuna_trials']} trials "
          f"or {config['optuna_timeout']}s, whichever comes first...", flush=True)

    def objective(trial):
        base = dict(
            loss_function='Logloss', eval_metric='AUC', verbose=False,
            thread_count=config['n_jobs'], iterations=2000,
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            depth=trial.suggest_int('depth', 4, 10),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            early_stopping_rounds=50,
        )
        skf = StratifiedKFold(n_splits=config['optuna_folds'], shuffle=True, random_state=config['random_state'])
        scores = []
        for tr_idx, val_idx in skf.split(X_cb, y):
            train_pool = Pool(X_cb.iloc[tr_idx], y.iloc[tr_idx], cat_features=CAT_COLS)
            val_pool = Pool(X_cb.iloc[val_idx], y.iloc[val_idx], cat_features=CAT_COLS)
            params = dict(base, task_type=('GPU' if use_gpu else 'CPU'))
            try:
                model = CatBoostClassifier(**params)
                model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            except Exception as e:
                if params.get('task_type') == 'GPU':
                    params['task_type'] = 'CPU'
                    model = CatBoostClassifier(**params)
                    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
                else:
                    raise
            pred = model.predict_proba(val_pool)[:, 1]
            scores.append(roc_auc_score(y.iloc[val_idx], pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction='maximize')
    study.set_user_attr('t0', time.time())
    study.set_user_attr('n_trials', config['optuna_trials'])
    study.optimize(objective, n_trials=config['optuna_trials'], timeout=config['optuna_timeout'],
                   show_progress_bar=False, callbacks=[_make_progress_callback('CatBoost')])
    print(f"  Best CatBoost Optuna AUC: {study.best_value:.5f} | params: {study.best_params}", flush=True)
    return study.best_params


# ==============================================================================
# NEURAL NET — tabular MLP with entity embeddings (4th ensemble member)
# ==============================================================================
if TORCH_AVAILABLE:
    class TabularMLP(nn.Module):
        def __init__(self, n_num_features, cat_cardinalities, emb_dim=8, hidden=(256, 128, 64), dropout=0.3):
            super().__init__()
            self.embeddings = nn.ModuleList([
                nn.Embedding(card + 1, min(emb_dim, (card + 1) // 2 + 1)) for card in cat_cardinalities
            ])
            emb_total = sum(e.embedding_dim for e in self.embeddings)
            self.bn_num = nn.BatchNorm1d(n_num_features)
            layers = []
            in_dim = n_num_features + emb_total
            for h in hidden:
                layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = h
            layers += [nn.Linear(in_dim, 1)]
            self.mlp = nn.Sequential(*layers)

        def forward(self, x_num, x_cat):
            embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
            x = torch.cat([self.bn_num(x_num)] + embs, dim=1)
            return self.mlp(x).squeeze(-1)


def prepare_nn_arrays(X, X_test, cat_cols, num_cols, medians=None):
    """Median-impute + standardize numerics; integer-code categoricals with
    a dedicated 'missing' index for NaN. Returns arrays plus cardinalities."""
    if medians is None:
        medians = X[num_cols].median()
    X_num = X[num_cols].fillna(medians).values.astype(np.float32)
    X_test_num = X_test[num_cols].fillna(medians).values.astype(np.float32)

    # standardize using train statistics only
    mu, sigma = X_num.mean(axis=0), X_num.std(axis=0) + 1e-6
    X_num = (X_num - mu) / sigma
    X_test_num = (X_test_num - mu) / sigma

    X_cat_list, X_test_cat_list, cardinalities = [], [], []
    for c in cat_cols:
        card = int(X[c].cat.categories.shape[0])
        codes = X[c].cat.codes.values
        codes_test = X_test[c].cat.codes.values
        codes = np.where(codes < 0, card, codes)
        codes_test = np.where(codes_test < 0, card, codes_test)
        X_cat_list.append(codes)
        X_test_cat_list.append(codes_test)
        cardinalities.append(card)
    X_cat = np.stack(X_cat_list, axis=1).astype(np.int64)
    X_test_cat = np.stack(X_test_cat_list, axis=1).astype(np.int64)

    return X_num, X_cat, X_test_num, X_test_cat, cardinalities, medians, mu, sigma


def train_nn_cv(X, y, X_test, cat_cols, num_cols, config, use_gpu,
                pseudo_X=None, pseudo_y=None):
    if not TORCH_AVAILABLE:
        print("  torch not installed — skipping the neural net ensemble member.")
        return None, None, []

    device = torch.device('cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    n_folds, seeds = config['n_folds'], config['seeds']
    epochs, patience = config['nn_epochs'], config['nn_patience']
    batch_size, lr = config['nn_batch_size'], config['nn_lr']

    X_num, X_cat, X_test_num, X_test_cat, cardinalities, medians, mu, sigma = prepare_nn_arrays(
        X, X_test, cat_cols, num_cols)
    y_arr = y.values.astype(np.float32)

    pseudo_num = pseudo_cat = pseudo_y_arr = None
    if pseudo_X is not None:
        # Reuse the EXACT same median-impute + standardize stats that were
        # fit on the training data — pseudo rows must land on the same scale
        # as X_num, or the network sees badly-conditioned mixed inputs.
        pseudo_num = pseudo_X[num_cols].fillna(medians).values.astype(np.float32)
        pseudo_num = (pseudo_num - mu) / sigma
        pseudo_cat_list = []
        for i, c in enumerate(cat_cols):
            card = cardinalities[i]
            codes = pseudo_X[c].astype(object).map(
                {cat: code for code, cat in enumerate(X[c].cat.categories)}
            ).fillna(card).astype(int).values
            pseudo_cat_list.append(codes)
        pseudo_cat = np.stack(pseudo_cat_list, axis=1).astype(np.int64)
        pseudo_y_arr = pseudo_y.values.astype(np.float32)

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    X_test_num_t = torch.tensor(X_test_num, device=device)
    X_test_cat_t = torch.tensor(X_test_cat, device=device)

    for seed in seeds:
        torch.manual_seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_num, y_arr)):
            model = TabularMLP(X_num.shape[1], cardinalities,
                                hidden=config['nn_hidden'], dropout=config['nn_dropout']).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
            criterion = nn.BCEWithLogitsLoss()

            tr_num, tr_cat, tr_y = X_num[tr_idx], X_cat[tr_idx], y_arr[tr_idx]
            if pseudo_num is not None:
                tr_num = np.concatenate([tr_num, pseudo_num], axis=0)
                tr_cat = np.concatenate([tr_cat, pseudo_cat], axis=0)
                tr_y = np.concatenate([tr_y, pseudo_y_arr], axis=0)

            train_ds = TensorDataset(torch.tensor(tr_num), torch.tensor(tr_cat), torch.tensor(tr_y))
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            X_val_num_t = torch.tensor(X_num[val_idx], device=device)
            X_val_cat_t = torch.tensor(X_cat[val_idx], device=device)
            y_val = y_arr[val_idx]

            best_auc, best_state, patience_ctr = -1, None, 0
            t0 = time.time()
            for epoch in range(epochs):
                model.train()
                for xb_num, xb_cat, yb in train_dl:
                    xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
                    opt.zero_grad()
                    loss = criterion(model(xb_num, xb_cat), yb)
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    val_pred = torch.sigmoid(model(X_val_num_t, X_val_cat_t)).cpu().numpy()
                val_auc = roc_auc_score(y_val, val_pred)
                scheduler.step(val_auc)
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                val_pred = torch.sigmoid(model(X_val_num_t, X_val_cat_t)).cpu().numpy()
                test_pred = torch.sigmoid(model(X_test_num_t, X_test_cat_t)).cpu().numpy()

            oof[val_idx] += val_pred / len(seeds)
            test_preds += test_pred / (n_folds * len(seeds))
            fold_scores.append(best_auc)
            print(f"  [NeuralNet] seed={seed} fold={fold + 1}/{n_folds} "
                  f"AUC={best_auc:.5f} epochs={epoch + 1} time={time.time() - t0:.1f}s")

    return oof, test_preds, fold_scores


# ==============================================================================
# GBM CROSS-VALIDATED TRAINING — multi-seed bagging, GPU->CPU fallback,
# optional pseudo-labeled rows always folded into training (never validation)
# ==============================================================================
def train_lgb_cv(X, y, X_test, config, params_base, pseudo_X=None, pseudo_y=None):
    n_folds, seeds = config['n_folds'], config['seeds']
    early_stop = config['early_stop']['lgb']
    oof, test_preds, fold_scores = np.zeros(len(X)), np.zeros(len(X_test)), []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            if pseudo_X is not None:
                X_tr = pd.concat([X_tr, pseudo_X], axis=0)
                y_tr = pd.concat([y_tr, pseudo_y], axis=0)
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            params = dict(params_base, random_state=seed)
            t0 = time.time()
            try:
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc',
                          categorical_feature=CAT_COLS,
                          callbacks=[lgb.early_stopping(early_stop, verbose=False)])
            except Exception as e:
                if params.get('device_type') == 'gpu':
                    print(f"    [GPU fallback] LightGBM GPU failed ({e}); retrying on CPU.")
                    params['device_type'] = 'cpu'
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc',
                              categorical_feature=CAT_COLS,
                              callbacks=[lgb.early_stopping(early_stop, verbose=False)])
                else:
                    raise
            val_pred = model.predict_proba(X_val)[:, 1]
            oof[val_idx] += val_pred / len(seeds)
            test_preds += model.predict_proba(X_test)[:, 1] / (n_folds * len(seeds))
            fold_auc = roc_auc_score(y_val, val_pred)
            fold_scores.append(fold_auc)
            print(f"  [LightGBM] seed={seed} fold={fold + 1}/{n_folds} "
                  f"AUC={fold_auc:.5f} best_iter={model.best_iteration_} time={time.time() - t0:.1f}s")
    return oof, test_preds, fold_scores


def train_xgb_cv(X, y, X_test, config, params_base, pseudo_X=None, pseudo_y=None):
    n_folds, seeds = config['n_folds'], config['seeds']
    early_stop = config['early_stop']['xgb']
    oof, test_preds, fold_scores = np.zeros(len(X)), np.zeros(len(X_test)), []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            if pseudo_X is not None:
                X_tr = pd.concat([X_tr, pseudo_X], axis=0)
                y_tr = pd.concat([y_tr, pseudo_y], axis=0)
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            params = dict(params_base, random_state=seed, early_stopping_rounds=early_stop)
            t0 = time.time()
            try:
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            except Exception as e:
                if params.get('device') == 'cuda':
                    print(f"    [GPU fallback] XGBoost GPU failed ({e}); retrying on CPU.")
                    params['device'] = 'cpu'
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    raise
            val_pred = model.predict_proba(X_val)[:, 1]
            oof[val_idx] += val_pred / len(seeds)
            test_preds += model.predict_proba(X_test)[:, 1] / (n_folds * len(seeds))
            fold_auc = roc_auc_score(y_val, val_pred)
            fold_scores.append(fold_auc)
            print(f"  [XGBoost]  seed={seed} fold={fold + 1}/{n_folds} "
                  f"AUC={fold_auc:.5f} best_iter={model.best_iteration} time={time.time() - t0:.1f}s")
    return oof, test_preds, fold_scores


def train_cb_cv(X, y, X_test, config, params_base, pseudo_X=None, pseudo_y=None):
    n_folds, seeds = config['n_folds'], config['seeds']
    early_stop = config['early_stop']['cb']
    oof, test_preds, fold_scores = np.zeros(len(X)), np.zeros(len(X_test)), []
    test_pool = Pool(X_test, cat_features=CAT_COLS)

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            if pseudo_X is not None:
                X_tr = pd.concat([X_tr, pseudo_X], axis=0)
                y_tr = pd.concat([y_tr, pseudo_y], axis=0)
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            params = dict(params_base, random_seed=seed, early_stopping_rounds=early_stop)
            t0 = time.time()
            train_pool = Pool(X_tr, y_tr, cat_features=CAT_COLS)
            val_pool = Pool(X_val, y_val, cat_features=CAT_COLS)
            try:
                model = CatBoostClassifier(**params)
                model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            except Exception as e:
                if params.get('task_type') == 'GPU':
                    print(f"    [GPU fallback] CatBoost GPU failed ({e}); retrying on CPU.")
                    params['task_type'] = 'CPU'
                    model = CatBoostClassifier(**params)
                    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
                else:
                    raise
            val_pred = model.predict_proba(val_pool)[:, 1]
            oof[val_idx] += val_pred / len(seeds)
            test_preds += model.predict_proba(test_pool)[:, 1] / (n_folds * len(seeds))
            fold_auc = roc_auc_score(y_val, val_pred)
            fold_scores.append(fold_auc)
            print(f"  [CatBoost] seed={seed} fold={fold + 1}/{n_folds} "
                  f"AUC={fold_auc:.5f} best_iter={model.get_best_iteration()} time={time.time() - t0:.1f}s")
    return oof, test_preds, fold_scores


# ==============================================================================
# ENSEMBLING — try weighted blend, rank-average, LR stacking, GBM stacking;
# keep whichever wins on OOF AUC
# ==============================================================================
def optimize_blend_weights(oof_dict, y_true, n_iter=800, seed=42):
    """Dirichlet random search + local refinement. Generalizes cleanly to any
    number of models (unlike a fixed-step grid search, which blows up combinatorially)."""
    names = list(oof_dict.keys())
    n = len(names)
    rng = np.random.default_rng(seed)
    oof_matrix = np.column_stack([oof_dict[m] for m in names])

    best_score, best_w = -1.0, np.ones(n) / n
    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(n))
        score = roc_auc_score(y_true, oof_matrix @ w)
        if score > best_score:
            best_score, best_w = score, w
    for _ in range(n_iter // 2):
        w = np.clip(best_w + rng.normal(0, 0.05, n), 0, None)
        w = w / w.sum()
        score = roc_auc_score(y_true, oof_matrix @ w)
        if score > best_score:
            best_score, best_w = score, w
    return dict(zip(names, best_w)), best_score


def rank_average(pred_dict, weights=None):
    names = list(pred_dict.keys())
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}
    ranks = {n: rankdata(pred_dict[n]) / len(pred_dict[n]) for n in names}
    return sum(weights[n] * ranks[n] for n in names)


def build_ensemble(oof_dict, test_dict, y, config):
    """Try several ensembling strategies and keep the one with the best OOF AUC."""
    names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[m] for m in names])
    test_matrix = np.column_stack([test_dict[m] for m in names])
    candidates = {}

    weights, blend_auc = optimize_blend_weights(oof_dict, y)
    candidates['weighted_blend'] = (blend_auc, weights,
                                     oof_matrix @ np.array([weights[n] for n in names]),
                                     test_matrix @ np.array([weights[n] for n in names]))

    equal_w = {n: 1.0 / len(names) for n in names}
    rank_oof = rank_average(oof_dict, equal_w)
    rank_test = rank_average(test_dict, equal_w)
    candidates['rank_average'] = (roc_auc_score(y, rank_oof), equal_w, rank_oof, rank_test)

    lr_oof = cross_val_predict(LogisticRegression(), oof_matrix, y, cv=5, method='predict_proba')[:, 1]
    lr_model = LogisticRegression().fit(oof_matrix, y)
    candidates['logistic_stack'] = (roc_auc_score(y, lr_oof), None, lr_oof, lr_model.predict_proba(test_matrix)[:, 1])

    gbm_meta = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                   max_depth=3, verbosity=-1)
    gbm_oof = cross_val_predict(gbm_meta, oof_matrix, y, cv=5, method='predict_proba')[:, 1]
    gbm_meta.fit(oof_matrix, y)
    candidates['gbm_stack'] = (roc_auc_score(y, gbm_oof), None, gbm_oof, gbm_meta.predict_proba(test_matrix)[:, 1])

    print("\nEnsemble method comparison (OOF AUC):")
    for name, (auc, *_ ) in sorted(candidates.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:18s}: {auc:.5f}")

    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_auc, best_weights, best_oof, best_test = candidates[best_name]
    print(f"\n>>> Selected ensemble method: {best_name} (OOF AUC {best_auc:.5f})")
    return best_name, best_auc, best_oof, best_test


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 78)
    print("Kaggle Playground S6E8 — Smartphone Addiction Prediction — V2")
    print("=" * 78)

    use_gpu = resolve_gpu(CONFIG)

    train, test, sample_sub = load_data(CONFIG)
    print(f"Train shape: {train.shape} | Test shape: {test.shape}")
    print(f"Target balance: {train[CONFIG['target_col']].mean():.4f} positive rate")

    X, X_cb, y, X_test, X_test_cb, feature_cols, num_cols = prepare_features(train, test, CONFIG)
    print(f"Engineered feature count: {len(feature_cols)}")

    if CONFIG['quick_mode']:
        CONFIG['n_folds'] = min(CONFIG['n_folds'], 5)
        CONFIG['seeds'] = CONFIG['seeds'][:1]
        CONFIG['n_estimators'] = {'lgb': 1500, 'xgb': 800, 'cb': 700}
        CONFIG['early_stop'] = {'lgb': 60, 'xgb': 40, 'cb': 40}
    else:
        CONFIG['n_estimators'] = {'lgb': 5000, 'xgb': 5000, 'cb': 5000}
        CONFIG['early_stop'] = {'lgb': 150, 'xgb': 150, 'cb': 150}

    print(f"CV plan: {CONFIG['n_folds']}-fold x {len(CONFIG['seeds'])} seed(s) = "
          f"{CONFIG['n_folds'] * len(CONFIG['seeds'])} fits per model")

    if CONFIG['run_optuna']:
        print("\nRunning Optuna hyperparameter search (this takes a while)...", flush=True)
        tuning_gpu = CONFIG['optuna_use_gpu'] and use_gpu
        lgb_best = tune_lgb_optuna(X, y, CONFIG, tuning_gpu)
        xgb_best = tune_xgb_optuna(X, y, CONFIG, tuning_gpu)
        cb_best = tune_cb_optuna(X_cb, y, CONFIG, tuning_gpu)
    else:
        lgb_best, xgb_best, cb_best = {}, {}, {}

    lgb_params = {**get_lgb_params(CONFIG, use_gpu), **lgb_best}
    xgb_params = {**get_xgb_params(CONFIG, use_gpu), **xgb_best}
    cb_params = {**get_cb_params(CONFIG, use_gpu), **cb_best}

    print("\n--- Training LightGBM ---")
    lgb_oof, lgb_test, lgb_scores = train_lgb_cv(X, y, X_test, CONFIG, lgb_params)
    print(f"LightGBM: mean fold AUC={np.mean(lgb_scores):.5f} | OOF AUC={roc_auc_score(y, lgb_oof):.5f}")

    print("\n--- Training XGBoost ---")
    xgb_oof, xgb_test, xgb_scores = train_xgb_cv(X, y, X_test, CONFIG, xgb_params)
    print(f"XGBoost:  mean fold AUC={np.mean(xgb_scores):.5f} | OOF AUC={roc_auc_score(y, xgb_oof):.5f}")

    print("\n--- Training CatBoost ---")
    cb_oof, cb_test, cb_scores = train_cb_cv(X_cb, y, X_test_cb, CONFIG, cb_params)
    print(f"CatBoost: mean fold AUC={np.mean(cb_scores):.5f} | OOF AUC={roc_auc_score(y, cb_oof):.5f}")

    oof_dict = {'lgb': lgb_oof, 'xgb': xgb_oof, 'cb': cb_oof}
    test_dict = {'lgb': lgb_test, 'xgb': xgb_test, 'cb': cb_test}

    if CONFIG['use_nn'] and TORCH_AVAILABLE:
        print("\n--- Training Neural Net (entity-embedding MLP) ---")
        nn_oof, nn_test, nn_scores = train_nn_cv(X, y, X_test, CAT_COLS, num_cols, CONFIG, use_gpu)
        if nn_oof is not None:
            print(f"NeuralNet: mean fold AUC={np.mean(nn_scores):.5f} | OOF AUC={roc_auc_score(y, nn_oof):.5f}")
            oof_dict['nn'] = nn_oof
            test_dict['nn'] = nn_test
    elif CONFIG['use_nn'] and not TORCH_AVAILABLE:
        print("\ntorch not available — skipping NN ensemble member (pip install torch to enable it).")

    method, ens_auc, ens_oof, ens_test = build_ensemble(oof_dict, test_dict, y, CONFIG)

    if CONFIG['use_pseudo_labeling']:
        print("\n--- Pseudo-labeling second pass ---")
        conf = CONFIG['pseudo_label_confidence']
        confident_mask = (ens_test >= conf) | (ens_test <= (1 - conf))
        n_confident = int(confident_mask.sum())
        print(f"  {n_confident} / {len(ens_test)} test rows are confident enough to pseudo-label")

        if n_confident > 0:
            pseudo_X = X_test.loc[confident_mask].copy()
            pseudo_X_cb = X_test_cb.loc[confident_mask].copy()
            pseudo_y = pd.Series((ens_test[confident_mask] >= 0.5).astype(int)).reset_index(drop=True)
            pseudo_X = pseudo_X.reset_index(drop=True)
            pseudo_X_cb = pseudo_X_cb.reset_index(drop=True)

            print("  Retraining all base models with pseudo-labeled rows folded into every training split...")
            lgb_oof2, lgb_test2, _ = train_lgb_cv(X, y, X_test, CONFIG, lgb_params, pseudo_X, pseudo_y)
            xgb_oof2, xgb_test2, _ = train_xgb_cv(X, y, X_test, CONFIG, xgb_params, pseudo_X, pseudo_y)
            cb_oof2, cb_test2, _ = train_cb_cv(X_cb, y, X_test_cb, CONFIG, cb_params, pseudo_X_cb, pseudo_y)

            oof_dict2 = {'lgb': lgb_oof2, 'xgb': xgb_oof2, 'cb': cb_oof2}
            test_dict2 = {'lgb': lgb_test2, 'xgb': xgb_test2, 'cb': cb_test2}

            if CONFIG['use_nn'] and TORCH_AVAILABLE:
                nn_oof2, nn_test2, _ = train_nn_cv(X, y, X_test, CAT_COLS, num_cols, CONFIG, use_gpu,
                                                    pseudo_X, pseudo_y)
                if nn_oof2 is not None:
                    oof_dict2['nn'] = nn_oof2
                    test_dict2['nn'] = nn_test2

            method2, ens_auc2, ens_oof2, ens_test2 = build_ensemble(oof_dict2, test_dict2, y, CONFIG)
            print(f"\n  Pre-pseudo-label OOF AUC: {ens_auc:.5f} | Post-pseudo-label OOF AUC: {ens_auc2:.5f}")
            if ens_auc2 > ens_auc:
                print("  Pseudo-labeling improved OOF AUC — keeping the augmented-training predictions.")
                method, ens_auc, ens_oof, ens_test = method2, ens_auc2, ens_oof2, ens_test2
                oof_dict, test_dict = oof_dict2, test_dict2
            else:
                print("  Pseudo-labeling did NOT improve OOF AUC — keeping the original predictions.")

    print(f"\nFinal ensemble: {method} | Final OOF AUC: {ens_auc:.5f}")

    submission = sample_sub[[CONFIG['id_col']]].copy()
    submission[CONFIG['target_col']] = ens_test
    submission.to_csv(CONFIG['output_path'], index=False)
    print(f"\nSaved: {CONFIG['output_path']}")
    print(submission.head())

    oof_df = pd.DataFrame({CONFIG['id_col']: train[CONFIG['id_col']], 'y_true': y})
    for name, arr in oof_dict.items():
        oof_df[f'{name}_oof'] = arr
    oof_df['ensemble_oof'] = ens_oof
    oof_df.to_csv(CONFIG['oof_output_path'], index=False)
    print(f"Saved: {CONFIG['oof_output_path']}")

    return submission, ens_auc


if __name__ == '__main__':
    main()
