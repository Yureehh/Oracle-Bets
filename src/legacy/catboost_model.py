# from dataclasses import dataclass

# import optuna
# import pandas as pd
# from catboost import CatBoostClassifier
# from sklearn.metrics import log_loss

# from prediction_models.gbdt_model import GradientBoostingModel
# from utils.logger import logger, models_logger
# from utils.paths import CATBOOST_BEST_HYPERPARAMETERS
# from utils.utils import load_model


# @dataclass
# class CatBoostModel(GradientBoostingModel):

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
#         model = CatBoostClassifier(**best_params)
#         model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

#         # Validate the model
#         self.validate_model(model, X_test, y_test, eval_gameids, eval_sides)

#         logger.info("Calculating and plotting feature importances...")
#         self.store_feature_importance(model, selected_features)
#         self.calculate_permutation_importance(model, X_test, y_test, selected_features)
#         self.calculate_and_plot_shap(model, X_train, selected_features)
#         logger.info("Finished calculating and plotting feature importances.\n")

#         return model

#     def get_best_hyperparameters(self, X_train, y_train, X_test, y_test):
#         if CATBOOST_BEST_HYPERPARAMETERS.exists():
#             best_params = load_model(CATBOOST_BEST_HYPERPARAMETERS)
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
#                 "iterations": 1000,
#                 "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#                 "depth": trial.suggest_int("depth", 4, 10),
#                 "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 3, 8),
#                 "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1.0),
#                 "border_count": trial.suggest_int("border_count", 32, 255),
#                 "loss_function": trial.suggest_categorical("loss_function", ["Logloss", "CrossEntropy"]),
#                 "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
#                 "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
#                 "used_ram_limit": "3gb",
#             }

#             if params["bootstrap_type"] == "Bayesian":
#                 params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
#             elif params["bootstrap_type"] == "Bernoulli":
#                 params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

#             clf = CatBoostClassifier(**params)
#             clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

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
