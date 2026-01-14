import os
import random
import itertools
import pandas as pd
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import xgboost as xgb

remove = [
 'occupation_j',
 'working_hours_j',
 'driving_license_j',
 'min_edu_eqf_j',
 'min_years_exp_j',
 'skills_req_j',
 'lang_skills_j',
 'permanent_j',
 'job_cat_c',
 'edu_eqf_c',
 'gender_c',
 'working_hours_c',
 'driving_license_c',
 'age_c',
 'years_exp_c',
 'skills_c',
 'lang_skills_c',
 'degree_rating_c',
 'permanent_c',
 'min_years_exp_int_j'
]

# Datsetes
DATASETS = {
    "HR_unbiased": "datasets/HR_unbiased/df_HR_unbiased.xls",
    "HR_biased": "datasets/HR_biased/df_HR_biased.xls",
    "MSLR-WEB10K": "datasets/MSLR-WEB10K/MSLR-WEB10K.xls",
    "Yahoo": "datasets/Yahoo/Yahoo.xls",
    "MQ2007": "datasets/MQ2007/MQ2007.xls"
}


OUTPUT_DIR = "benchmark_final_results_csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Parametri per combinazioni Random Search
param_space = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 7, 9],
    'n_estimators': [50, 100, 150, 200],
    'num_leaves': [20, 31, 40, 50],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'lambdarank_truncation_level': [3, 5, 7, 10]
}

param_keys = list(param_space.keys())
num_trials = 10

combinations = list(itertools.product(*param_space.values()))
random.shuffle(combinations)
sampled_combinations = combinations[:num_trials]


def split_by_group(values, group_sizes):
    out, i = [], 0
    for g in group_sizes:
        out.append(values[i:i + g])
        i += g
    return out


for dataset_name, dataset_path in DATASETS.items():
    print(f"[INFO] Starting benchmark on dataset: {dataset_name}")

    # Caricamento dataset
    df = pd.read_csv(dataset_path, sep=",", encoding="utf-8-sig")
    qid_col, target_col = "qid", "relevance"

    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42, stratify=df[qid_col])
    print(f"[INFO] Loaded dataset {dataset_name}: Train={len(df_train)}, Validation={len(df_val)}")

    # Raggruppamento query
    group_train = df_train.groupby(qid_col).size().to_numpy()
    group_val = df_val.groupby(qid_col).size().to_numpy()

    if dataset_name in ("HR_unbiased", "HR_biased"):
        
        X_train = df_train.drop(columns=[qid_col, target_col] + remove)
        y_train = df_train[target_col].values
        X_val = df_val.drop(columns=[qid_col, target_col] + remove)
        y_val = df_val[target_col].values

    else:
        X_train = df_train.drop(columns=[qid_col, target_col])
        y_train = df_train[target_col].values
        X_val = df_val.drop(columns=[qid_col, target_col])
        y_val = df_val[target_col].values

    # Dataset LightGBM
    train_data_lgb = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data_lgb)

    results = []
    qids_val = df_val[qid_col].drop_duplicates().tolist()

    for i, combo in enumerate(tqdm(sampled_combinations, desc=f"Random search - {dataset_name}"), start=1):
        random_params = dict(zip(param_keys, combo))
        start_time = time.time()

        models_to_train = [
            {'name': 'Pointwise', 'params': {'objective': 'regression', 'verbosity': -1, **random_params}},
            {'name': 'Pairwise', 'params': {'objective': 'lambdarank', 'metric': 'ndcg', 'verbosity': -1, **random_params}},
            {'name': 'LambdaMART', 'params': {'objective': 'lambdarank', 'metric': 'ndcg', 'verbosity': -1, **random_params}},
            {'name': 'XGBRanker', 'params': {
                'objective': 'rank:ndcg',
                'eval_metric': 'ndcg',
                'verbosity': 0,
                'eta': random_params['learning_rate'],
                'max_depth': random_params['max_depth'],
                'n_estimators': random_params['n_estimators'],
                'subsample': random_params['subsample'],
                'colsample_bytree': random_params['colsample_bytree']
            }}
        ]

        for model_spec in models_to_train:
            model_name = model_spec['name']
            params = model_spec['params']
            success = True

            try:
                if model_name != 'XGBRanker':
                    model = lgb.train(params, train_data_lgb, num_boost_round=random_params['n_estimators'])
                else:
                    model = xgb.sklearn.XGBRanker(**params)
                    model.fit(X_train, y_train, group=group_train,
                              eval_set=[(X_val, y_val)], eval_group=[group_val], verbose=False)
            except Exception as e:
                print(f"[ERROR] {model_name} failed: {e}")
                success = False
                continue

            elapsed = round(time.time() - start_time, 2)

            # Calcolo NDCG@5 per ogni query
            y_pred = model.predict(X_val)
            y_true_grouped = split_by_group(y_val, group_val)
            y_pred_grouped = split_by_group(y_pred, group_val)

            ndcg_by_query = []
            for q_idx in range(len(group_val)):
                try:
                    ndcg_q = ndcg_score([y_true_grouped[q_idx]], [y_pred_grouped[q_idx]], k=5)
                except Exception:
                    ndcg_q = np.nan
                ndcg_by_query.append(round(ndcg_q, 5))

            mean_ndcg = np.nanmean(ndcg_by_query)

            # Salvataggio risultati in formato wide
            result_entry = {
                "dataset": dataset_name,
                "model": model_name,
                "trial": i,
                "elapsed_time_sec": elapsed,
                "success": success,
                "mean_ndcg@5": round(mean_ndcg, 5),
                **random_params
            }

            # Aggiunge una colonna per ogni query
            for q_idx, qid_val in enumerate(qids_val):
                result_entry[f"qid_{qid_val}_ndcg@5"] = ndcg_by_query[q_idx] if q_idx < len(ndcg_by_query) else np.nan

            results.append(result_entry)

    # Salvataggio
    df_results = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, f"benchmark_{dataset_name}_models.csv")
    df_results.to_csv(out_path, index=False, encoding="utf-8-sig")