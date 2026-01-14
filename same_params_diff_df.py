import os
import random
import itertools
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
from run_experiments import run_experiment, log
from sklearn.model_selection import train_test_split

DATA_PATH = "datasets/HR_unbiased/df_HR_unbiased.xls" #[HR/HR.xls ; MSLR-WEB10K/MSLR-WEB10K.xls ; Yahoo/Yahoo ; MQ2007/MQ2007.xls]


df = pd.read_csv(DATA_PATH, sep=",", encoding="utf-8-sig")
feature_names = [c for c in df.columns if c.startswith("fitness_")]
remove = [c for c in df.columns if c not in feature_names + ['id_c']]
df_train, df_val = train_test_split(df, test_size=0.3, random_state=42, stratify = df["qid"])
print(f"[INFO] Loaded dataset: Train={len(df_train)}, Val={len(df_val)}")

param_grid = {
    'max_depth_shallow': [2, 3, 4],
    'compare_candidates': [True, False],
    'neigh': [3, 4, 5],
    'ndcg': [5, 10, 15],
    'distance' : ['None', 'euclidean'], 
    'optimize': [True, False],
    'leave_two_out': [True, False],
    'validation': [0.2, 0.3, 0.4],
    'optimization_gain': [0.05, 0.03, 0.01],
    'max_depth_pdt': [2, 3, 4]
}

num_trials = 1 # numero di cicli
results_all = []

# Combinazioni possibili 
combinations = list(itertools.product(
    param_grid['max_depth_shallow'],
    param_grid['compare_candidates'],
    param_grid['neigh'],
    param_grid['ndcg'],
    param_grid['distance'], 
    param_grid['optimize'],
    param_grid['leave_two_out'], 
    param_grid['validation'],
    param_grid['optimization_gain'],
    param_grid['max_depth_pdt']
))

random.shuffle(combinations)
sampled_combinations = combinations[:num_trials]

os.makedirs("datasets/HR_unbiased", exist_ok=True)  #[HR ; MSLR-WEB10K ; Yahoo ; MQ2007]
output_path = os.path.join("datasets/HR_unbiased", "random_search_results_HR_unbiased_1_experiment_2.csv")

print(f"Starting random search with {len(sampled_combinations)} combinations...")

for i, combo in enumerate(tqdm(sampled_combinations, desc="Random Search Progress"), start=1):
    (  max_depth_shallow,
        compare_candidates,
        neigh,
        ndcg_val,
        distance, 
        optimize,
        leave_two_out, 
        validation,
        optimization_gain,
        max_depth_pdt
    ) = combo

    print(f"[{datetime.now():%H:%M:%S}] Trial {i}/{len(sampled_combinations)}")

    config = {
    "qid_col": 'qid',
    "target_col": 'relevance',
    "columns_to_remove": ["leaf_shallow_tree", "residuals", "predicted", "qid", "relevance"] + remove ,
    "xai_max_depth": 2,  # max_depth_shallow
    "pdt_params": {
        "compare_candidates": False,      # compare_candidates
        "optimize": False,                # optimize
        "neighbours": 4,                 # neigh
        "max_depth": 2,                  # max_depth_pdt
        "distance": "euclidean",         # distance
        "leave_two_out": False,          # leave_two_out
        "validation_split": 0.2,         # validation
        "optimization_gain": 0.01,       # optimization_gain
        "ndcgat": 5                      # ndcg_val
    },
    }

    start_time = time.time()
    try:
        trial_result = run_experiment(config=config, df_train = df_train, df_val = df_val, verbose=False)
    except Exception as e:
        print(f"[ERROR] Trial {i}/{len(sampled_combinations)} failed: {e}")
        continue
    end_time = time.time()


    for res in trial_result:  # 5 righe per esperimento
        res["trial"] = i
        res["elapsed_time_sec"] = round(end_time - start_time, 2)
        res["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_all.append(res)

    # # Campi di controllo
    # trial_result["trial"] = i
    # trial_result["elapsed_time_sec"] = round(end_time - start_time, 2)
    # trial_result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # results_all.append(results_all)

    # Salvataggio incrementale
    #if i % 10 == 0:
      #  df_partial = pd.DataFrame(results_all)
       # df_partial.to_csv(output_path, index=False)
       # print(f"[INFO] Partial save after {i} trials {output_path}")

df_results = pd.DataFrame(results_all)
df_results.to_csv(output_path, index=False)
print(f"Random search completed. {len(df_results)} experiments saved in: {output_path}")