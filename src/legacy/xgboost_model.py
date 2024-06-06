# from dataclasses import dataclass

# import optuna
# import pandas as pd
# from sklearn.metrics import log_loss
# from xgboost import XGBClassifier

# from prediction_models.gbdt_model import GradientBoostingModel
# from utils.logger import logger, models_logger
# from utils.paths import XGBOOST_BEST_HYPERPARAMETERS
# from utils.utils import load_model


# @dataclass
# class XGBoostModel(GradientBoostingModel):

#     def train_model(self):
#         columns_to_drop = ["gameid", "side"]
#         self.training_data.sort_values(by=["date", "gameid", "side"], inplace=True)
#         X, y = self.training_data.drop(["date", "result"], axis=1), self.training_data["result"]

#         # Categorical columns handling
#         X, categorical_cols = self.preprocess_categorical_features(X, exclude_cols=columns_to_drop)

#         # Split the data into training and testing sets
#         X_train, X_test, y_train, y_test = self.grouped_train_test_split(X, y, X["gameid"], test_size=0.25)
#         eval_gameids, eval_sides = X_test["gameid"], X_test["side"]

#         # Drop specific columns
#         X_train = X_train.drop(columns=columns_to_drop, inplace=False, errors="ignore")
#         X_test = X_test.drop(columns=columns_to_drop, inplace=False, errors="ignore")

#         # Process likelihood columns and fuse opposing team features
#         X_train = GradientBoostingModel.process_players_likelihood_columns(X_train)
#         X_test = GradientBoostingModel.process_players_likelihood_columns(X_test)
#         X_train = GradientBoostingModel.fuse_opposing_team_features(X_train)
#         X_test = GradientBoostingModel.fuse_opposing_team_features(X_test)

#         # Remove unnecessary columns
#         X_train = GradientBoostingModel.remove_unnecessary_columns(X_train)
#         selected_features = X_train.columns
#         X_test = X_test[selected_features]

#         # Update and store categorical features
#         categorical_features = [col for col in categorical_cols if col in selected_features]
#         self.store_model_features(selected_features)
#         self.store_categorical_features(categorical_features)

#         # Get best hyperparameters and fit the model
#         best_params = self.get_best_hyperparameters(X_train, y_train, X_test, y_test)
#         model = XGBClassifier(**best_params, enable_categorical=False)
#         model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

#         # Validate the model
#         self.validate_model(model, X_test, y_test, eval_gameids, eval_sides)

#         logger.info("Calculating and plotting feature importances...")
#         self.store_feature_importance(model, selected_features)
#         self.calculate_permutation_importance(model, X_test, y_test, selected_features)
#         self.calculate_and_plot_shap(model, X_train, selected_features)
#         logger.info("Finished calculating and plotting feature importances.\n")

#         return model

#     def get_best_hyperparameters(self, X_train, y_train, X_test, y_test):
#         if XGBOOST_BEST_HYPERPARAMETERS.exists():
#             best_params = load_model(XGBOOST_BEST_HYPERPARAMETERS)
#             logger.info(f"Found best hyperparameters: {best_params}\n")
#             models_logger.info(f"Found best hyperparameters: {best_params}\n")
#         else:
#             best_params = self.optimize_hyperparameters(X_train, y_train, X_test, y_test)
#         return best_params

#     def optimize_hyperparameters(
#         self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
#     ) -> dict:
#         def objective(trial):
#             params = {
#                 "verbosity": 0,
#                 "objective": "binary:logistic",
#                 "tree_method": "exact",
#                 "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
#                 "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
#                 "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
#                 "subsample": trial.suggest_float("subsample", 0.2, 1.0),
#                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
#                 "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
#             }

#             if params["booster"] in ["gbtree", "dart"]:
#                 params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
#                 params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
#                 params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
#                 params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
#                 params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

#             if params["booster"] == "dart":
#                 params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
#                 params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
#                 params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
#                 params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

#             clf = XGBClassifier(**params, enable_categorical=False)
#             clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

#             pred_proba = clf.predict_proba(X_test)[:, 1]
#             logloss_score = log_loss(y_test, pred_proba)

#             return logloss_score

#         study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
#         study.optimize(objective, n_trials=self.trials)
#         self.store_best_hyperparameters(study)
#         logger.info(f"Best hyperparameters: {study.best_params}")
#         logger.info(f"Best log loss: {study.best_value:.4f}\n")
#         models_logger.info(f"Best hyperparameters: {study.best_params}")
#         models_logger.info(f"Best log loss: {study.best_value:.4f}\n")
#         return study.best_params
