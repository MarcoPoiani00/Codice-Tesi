import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'PDT')))

from Rule_Tree_2.tree.PairwiseDistanceTreeRegressor import PairwiseDistanceTreeRegressor 
from util_exp_funct import df_pairwise
from util_exp_funct import concat_for_prediction
from collections import defaultdict
import numpy as np
from sklearn.metrics import ndcg_score
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

class LetorPDT:
    def __init__(self, qid, relevance, columns_to_remove: list, compare_candidates = False, 
                 optimize = True, neighbours = 3, max_depth = 4 , distance = 'None', leave_two_out = False, 
                 validation_split = 0.2, optimization_gain = 0.01, ndcgat = 5):  
        self.compare_candidates = compare_candidates
        self.neighbours = neighbours
        self.max_depth = max_depth
        self.distance = distance
        self.ndcgat = ndcgat
        self.qid = qid
        self.relevance = relevance
        self.columns_to_remove = columns_to_remove
        self.optimize = optimize
        self.leave_two_out = leave_two_out
        self.validation_split = validation_split
        self.optimization_gain = optimization_gain
        self.pdt_models = {}
        self.train_matrix_dict = {}
        self.train_matrix_sign_dict = {}
        self.best_depths = {} 

        if self.distance == 'euclidean':
            self.compare_candidates = False

    # Alleno il PDT ad una profondità inserita come parametro, poi inizio il processo di ottimizzazione
    def _train_pdt_at_depth(self, X_train_pw, y_train_pairwise, X_train_pairwise, feature_names, depth):
        params_model = {'max_depth': depth, 'min_samples_leaf': 1, 'min_samples_split': 2, 'random_state': 42,
                        'fix_feature': True, 'fix_threshold': False}
        
        PDT = PairwiseDistanceTreeRegressor(**params_model)
        PDT.fit(X_train_pw, y_train_pairwise, X_train_pairwise)
        _, train_matrix = concat_for_prediction(X_train_pw, X_train_pw, feature_names, PDT, consider_abs_diff=True)
        train_matrix_sign = np.where(train_matrix  >= 0, 0, 1)
        return PDT, train_matrix, train_matrix_sign

    # La valutazione sfrutta la logica del predict
    def _evaluate_leaf(self, df_val, residuals_train, tree_score, distance_matrix, distance_sign):
        n_test = distance_matrix.shape[0]
        final_scores = []

        for idx_test in range(n_test):
            distances = np.abs(distance_matrix[idx_test])
            nearest_idx = np.argsort(distances)[:self.neighbours]

            # Calcola correzione basata sui residui dei vicini (come in predict)
            if self.compare_candidates:
                bonus, malus = [], []
                for j in nearest_idx:
                    sign =  distance_sign[idx_test][j]
                    residual = residuals_train[j]
                    if sign == 0:
                        bonus.append(residual)
                    else:
                        malus.append(residual)
                correction = (np.mean(bonus) if bonus else 0) - (np.mean(malus) if malus else 0)
            else:
                valid_residuals = residuals_train[nearest_idx]
                correction = np.mean(valid_residuals) if len(valid_residuals) > 0 else 0.0

            final_scores.append(tree_score + correction)

        # Calcolo NDCG solo se ci sono almeno due record
        if len(df_val) > 1:
            ndcg_val = ndcg_score(
                [df_val[self.relevance].values],
                [np.array(final_scores)],
                k=self.ndcgat
            )
            return ndcg_val

        return -float("inf")
        
    def _optimize_depth(self, X_train_pw, y_train_pairwise, X_train_pairwise, feature_names, df_wk, residuals_list, tree_score, query_leaf):
        n = len(df_wk)

        # Caso foglia troppo piccola
        if n < 4:
            PDT, train_matrix, train_matrix_sign = self._train_pdt_at_depth(
                X_train_pw, y_train_pairwise, X_train_pairwise, feature_names, self.max_depth
            )
            return PDT, train_matrix, train_matrix_sign, self.max_depth, -float("inf")

        best_depth = self.max_depth
        best_score = -float("inf")
        current_depth = self.max_depth
        improved = True

        while improved:
            improved = False
            depths = [d for d in [current_depth - 1, current_depth, current_depth + 1] if d >= 1]

            for depth in depths:
                if n < 10 and self.leave_two_out:
                    ndcg_scores = []

                    for i in range(n):
                        for j in range(i + 1, n):
                            val_idx = [i, j]
                            train_idx = [k for k in range(n) if k not in val_idx]
                            if len(train_idx) < 2:
                                continue

                            # Subset training
                            X_train_pw_sub = X_train_pw[train_idx]
                            X_train_pairwise_sub = X_train_pairwise[train_idx]
                            y_train_pairwise_sub = y_train_pairwise[train_idx]

                            PDT, _, _ = self._train_pdt_at_depth(
                                X_train_pw_sub, y_train_pairwise_sub, X_train_pairwise_sub, feature_names, depth
                            )

                            # Validation: 2 record esclusi
                            X_val_pw = X_train_pw[val_idx]
                            _, val_matrix = concat_for_prediction(X_val_pw, X_train_pw_sub, feature_names, PDT, consider_abs_diff=True)
                            val_matrix_sign = np.where(val_matrix >= 0, 0, 1)

                            df_val = df_wk.iloc[val_idx].reset_index(drop=True)
                            residuals_train = df_wk.iloc[train_idx]['residuals'].values

                            ndcg = self._evaluate_leaf(df_val, residuals_train, tree_score, val_matrix, val_matrix_sign)
                            ndcg_scores.append(ndcg)

                    ndcg_leaf = np.mean(ndcg_scores) if ndcg_scores else -float("inf")

                # Caso dataset più grande o LOT non richiesto
                else:
                    df_train_wk, _ = train_test_split(df_wk, test_size=self.validation_split, random_state=42)
                    train_idx = df_train_wk.index.to_numpy()

                    # Train PDT SOLO su training 
                    X_train_pw_sub = X_train_pw[train_idx]
                    X_train_pairwise_sub = X_train_pairwise[train_idx]
                    y_train_pairwise_sub = y_train_pairwise[train_idx]

                    PDT, train_matrix, train_matrix_sign = self._train_pdt_at_depth(
                        X_train_pw_sub, y_train_pairwise_sub, X_train_pairwise_sub, feature_names, depth
                    )

                    # Valuto su TUTTO il dataset (val + train)
                    X_full_pw = df_wk.drop(self.columns_to_remove, axis=1).values
                    _, full_matrix = concat_for_prediction(X_full_pw, X_train_pw_sub, feature_names, PDT, consider_abs_diff=True)
                    full_matrix_sign = np.where(full_matrix >= 0, 0, 1)

                    residuals_train = df_train_wk['residuals'].values
                    ndcg_leaf = self._evaluate_leaf(df_wk, residuals_train, tree_score, full_matrix, full_matrix_sign)
                    
                if ndcg_leaf - best_score > self.optimization_gain:
                    best_score = ndcg_leaf
                    best_depth = depth
                    improved = True

            current_depth = best_depth
            
        PDT, train_matrix , train_matrix_sign = self._train_pdt_at_depth(
        X_train_pw, y_train_pairwise, X_train_pairwise, feature_names, best_depth)

        return PDT, train_matrix, train_matrix_sign, best_depth, best_score

    def fit(self, df, leaves, query, feature_names):
        for q in query:
            for l in leaves:
                df_wk = df[(df[self.qid] == q) & (df['leaf_shallow_tree'] == l)].copy().reset_index(drop=True)
                query_leaf = f"{q}_{l}"

                if len(df_wk) >= 2: 
                    X_train_pw = df_wk.drop(self.columns_to_remove, axis=1).values
    
                    # Aggiungo modularità al modo in cui voglio misurare la distanza tra i record
                    if self.distance == 'euclidean':
                        X_train_matrix = pairwise_distances(X_train_pw)
                    else: 
                        score = np.asarray(df_wk[self.relevance]).reshape(-1, 1)
                        X_train_matrix = score - score.T
                    
                    df_X_train = df_pairwise(X_train_pw, feature_names, X_train_matrix, consider_abs_diff=True)
                    df_X_train.rename(columns={'overall_euclidean_distance_sklr': 'distance'}, inplace=True)
                    
                    new_feature_names = list(df_X_train.drop(columns=['indexes', 'distance']).columns)
                    X_train_pairwise = df_X_train[new_feature_names].values
                    y_train_pairwise = df_X_train['distance'].values
                    
                    residuals_list = df_wk['residuals'].values
                    tree_score = df_wk['predicted'].iloc[0]
    
    
                    if self.optimize:
                        best_model, best_matrix, best_sign, best_depth, best_score = self._optimize_depth(
                            X_train_pw, y_train_pairwise, X_train_pairwise, feature_names, df_wk, residuals_list, tree_score, query_leaf)
                        
                    else:
                        best_model, best_matrix, best_sign = self._train_pdt_at_depth(
                            X_train_pw, y_train_pairwise, X_train_pairwise, feature_names, self.max_depth)
                        
                        best_depth = self.max_depth
                        best_score = None
                
                    self.pdt_models[query_leaf] = best_model
                    self.train_matrix_dict[query_leaf] = best_matrix
                    self.train_matrix_sign_dict[query_leaf] = best_sign
                    self.best_depths[query_leaf] = best_depth
                
                else:
                    self.pdt_models[query_leaf] = None
                    self.train_matrix_dict[query_leaf] = None
                    self.train_matrix_sign_dict[query_leaf] = None
                    self.best_depths[query_leaf] = 0 # o None
            
    def predict(self, df, leaves, query, df_train, feature_names):
        all_df = []
        for q in query:
            for l in leaves:
                df_wk = df[(df[self.qid] == q) & (df['leaf_shallow_tree'] == l)].copy().reset_index(drop=True)
                query_leaf = f"{q}_{l}"
                if len(df_wk) == 0:
                    continue
                
                # Recupera i dati di training per la stessa query e foglia
                df_train_wk = df_train[(df_train[self.qid] == q) & (df_train['leaf_shallow_tree'] == l)].copy().reset_index(drop=True)
                if df_train_wk.empty:
                    continue

                model = self.pdt_models.get(query_leaf, None)
                if model is None:
                    continue

                test_columns = [col for col in self.columns_to_remove if col != 'residuals']
                X_train_pw = df_train_wk.drop(self.columns_to_remove, axis=1).values
                X_test_pw = df_wk.drop(test_columns, axis=1).values
                residuals_list = df_train_wk['residuals'].values
                tree_score = df_wk['predicted'].iloc[0]

                # Calcola matrice di distanza test–train 
                _, test_train_matrix = concat_for_prediction(
                    X_test_pw, X_train_pw, feature_names, model, consider_abs_diff=True
                )

                # Per ogni nuovo candidato, trova i k vicini più simili nel training
                final_scores = []
                final_corrections = []

                for candidate in range(len(df_wk)):
                    distances = np.abs(test_train_matrix[candidate])
                    neigh = np.argsort(distances)[:self.neighbours]
                    # Calcola la correzione in base ai residui dei vicini
                    if self.compare_candidates:
                        bonus, malus = [], []
                        for j in neigh:
                            sign = (test_train_matrix[candidate][j] < 0).astype(int)
                            residual = residuals_list[j]
                            if sign == 0:
                                bonus.append(residual)
                            else:
                                malus.append(residual)
                        correction = (np.mean(bonus) if bonus else 0) - (np.mean(malus) if malus else 0)
                    else:
                        valid_residuals = residuals_list[neigh]
                        correction = np.mean(valid_residuals) if len(valid_residuals) > 0 else 0.0

                    final_score = tree_score + correction
                    final_corrections.append(correction)
                    final_scores.append(final_score)

                # Aggiunge risultati per la foglia corrente
                df_wk['correction_final'] = final_corrections
                df_wk['score_final'] = final_scores
                df_wk['query_leaf'] = query_leaf
                all_df.append(df_wk)

        if not all_df:
            return pd.DataFrame()

        final_df = pd.concat(all_df, ignore_index=True)
        return final_df

    def score(self, df_pred, ndcg_):
        scores_per_leaf = {}
        scores_per_query = {}

        # Valutazione per foglia
        for query_leaf in df_pred['query_leaf'].unique():
            df_leaf = df_pred[df_pred['query_leaf'] == query_leaf]
            if len(df_leaf) > 1:
                score_leaf = ndcg_score([df_leaf[self.relevance].values], [df_leaf['score_final'].values], k=ndcg_)
                scores_per_leaf[query_leaf] = round(score_leaf, 4)

        # Valutazione per query
        for q in df_pred[self.qid].unique():
            df_q = df_pred[df_pred[self.qid] == q]
            if len(df_q) > 1:
                score_query = ndcg_score([df_q[self.relevance].values], [df_q['score_final'].values], k=ndcg_)
                scores_per_query[str(q)] = round(score_query, 4)

        return scores_per_leaf, scores_per_query