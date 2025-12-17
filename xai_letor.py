import pandas as pd
from sklearn.preprocessing import LabelEncoder
from RuleTree.tree.RuleTreeRegressor import RuleTreeRegressor

class XAILetor:

    def __init__(self, qid, feature_names, target_column, max_depth = 2):
        self.qid = qid
        self.feature_names = feature_names
        self.target_column = target_column
        self.max_depth = max_depth
        self.models = {}
        self.label_encoders = {}

    def fit(self, df):
        query_groups = df.groupby(self.qid)
        all_group_dfs = []

        for query_id, group_df in query_groups:
            X_train = group_df.loc[:, self.feature_names].values
            y_train = group_df.loc[:, self.target_column].values
        
            model = RuleTreeRegressor(
                max_depth= self.max_depth,
                criterion='squared_error',
                prune_useless_leaves=True,
                random_state=43
            )
            model.fit(X_train, y_train)
            predicted = model.predict(X_train)

            self.models[query_id] = model
            
            labels = model.apply(X_train)
            le = LabelEncoder()
            encoded_labels = le.fit_transform(labels) + 1
            self.label_encoders[query_id] = le
    
            group_df = group_df.copy()
            group_df['leaf_shallow_tree'] = encoded_labels
            group_df['predicted'] = predicted
            group_df['residuals'] = group_df[self.target_column] - group_df['predicted']
            all_group_dfs.append(group_df)  
    
        final_df = pd.concat(all_group_dfs, ignore_index=True)
        return self.models, self.label_encoders, final_df

    def predict(self, df_predict):
        predicted_dfs = []
        query_groups_predict = df_predict.groupby(self.qid)

        for query_id, group_df_predict in query_groups_predict:
            model = self.models[query_id]
            X_predict = group_df_predict.loc[:, self.feature_names].values
    
            pred = model.predict(X_predict)
            labels_predict = model.apply(X_predict)
            le_predict = self.label_encoders[query_id]
            encoded_labels_predict = le_predict.transform(labels_predict.ravel()) + 1
            
            group_df_predict = group_df_predict.copy()
            group_df_predict['leaf_shallow_tree'] = encoded_labels_predict
            group_df_predict['predicted'] = pred
            predicted_dfs.append(group_df_predict)

        final_df_predict = pd.concat(predicted_dfs, ignore_index=True)
        return final_df_predict