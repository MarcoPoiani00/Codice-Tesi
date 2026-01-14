import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xai_letor import XAILetor
from Letor_PDT import LetorPDT
from datetime import datetime

def log(msg: str, verbose=True):
    if verbose:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] {msg}")

def run_experiment(config, df_train, df_val, ndcg = 5, verbose=True):
    feature_names = [c for c in df_train.columns if c not in [config["qid_col"], config["target_col"]] or c.startswith("fitness_")]
    #feature_names = [c for c in df_train.columns if c.startswith("fitness_")]

    # XAI_LETOR
    log("Fitting XAI_LETOR...", verbose)
    xai_model = XAILetor(
        qid=config["qid_col"],
        feature_names=feature_names,
        target_column=config["target_col"],
        max_depth=config["xai_max_depth"]
    )
    _, _, df_transformed = xai_model.fit(df_train)
    log("XAI_LETOR fit completed.", verbose)

    # LETOR_PDT
    log("Fitting LETOR_PDT...", verbose)
    query_list = df_transformed[config["qid_col"]].unique().tolist()
    leaves_list = df_transformed["leaf_shallow_tree"].unique().tolist()
    feature_names_pdt = [c for c in df_transformed.columns if c not in config["columns_to_remove"]]

    pdt_model = LetorPDT(
        qid=config["qid_col"],
        relevance=config["target_col"],
        columns_to_remove=config["columns_to_remove"],
        **config["pdt_params"]
    )
    pdt_model.fit(df_transformed, leaves_list, query_list, feature_names_pdt)
    log("LETOR_PDT fit completed.", verbose)

    # Inference
    log("Predictions...", verbose)
    df_val_transformed = xai_model.predict(df_val)
    results_multi = []

    for ndcg_val in [5, 10, 15, 20, 25]:
        df_pred_val = pdt_model.predict(
            df_val_transformed, leaves_list, query_list, df_transformed, feature_names_pdt
        )
        scores_leaf, scores_query = pdt_model.score(df_pred_val, ndcg_=ndcg_val)
         # Count number of documents per query in validation
        query_counts = df_pred_val[config["qid_col"]].value_counts().to_dict()
        query_counts_str = {str(k): int(v) for k, v in query_counts.items()}

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "xai_max_depth": xai_model.max_depth,
            **config["pdt_params"],
            "ndcg_at": ndcg_val,
            "mean_ndcg_query": np.mean(list(scores_query.values())) if scores_query else 0,
            "mean_ndcg_leaf": np.mean(list(scores_leaf.values())) if scores_leaf else 0,
        }

        # Add query-level NDCG and record count
        for q, ndcg_q in scores_query.items():
            result[f"ndcg_query_{q}"] = ndcg_q
            result[f"records_query_{q}"] = query_counts_str.get(q, 0)  # add record count for that query

        # Add leaf-level NDCG
        for leaf, ndcg_l in scores_leaf.items():
            result[f"ndcg_leaf_{leaf}"] = ndcg_l

        results_multi.append(result)

    log("Experiment completed successfully.", verbose)
    return results_multi



    
    # df_pred_val = pdt_model.predict(df_val_transformed, leaves_list, query_list, df_transformed, feature_names_pdt)
    # scores_leaf, scores_query = pdt_model.score(df_pred_val, ndcg_= ndcg)

    # # Collect results
    # result = {
    #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #     "xai_max_depth": xai_model.max_depth,
    #     **config["pdt_params"],
    #     "mean_ndcg_query": np.mean(list(scores_query.values())) if scores_query else 0,
    #     "mean_ndcg_leaf": np.mean(list(scores_leaf.values())) if scores_leaf else 0,
    # }

    # # Aggiunge dettagli per query e leaf 
    # for q, ndcg in scores_query.items():
    #     result[f"ndcg_query_{q}"] = ndcg
    # for leaf, ndcg in scores_leaf.items():
    #     result[f"ndcg_leaf_{leaf}"] = ndcg

    # log("Experiment completed successfully.", verbose)
    # return result