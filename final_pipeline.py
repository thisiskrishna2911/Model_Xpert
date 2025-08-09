import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error, mean_absolute_error
import joblib, optuna
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# Global DataFrame to store model results
global_model_results = pd.DataFrame(columns=[
    'model_name', 'algorithm', 'accuracy', 'f1_score',
    'classification_report', 'hyperparameters', 'model_file'
])

global_model_results2 = pd.DataFrame(columns=[
    # Common columns for all models
    'model_name',               # Name of the model (e.g., algorithm_trial_number)
    'algorithm',                # Algorithm used (e.g., SVR, RandomForest, etc.)
    'model_file',               # File where the model is saved
    'hyperparameters',          # Dictionary of hyperparameters used in the model



    # Regression-specific metrics
    'mse',                      # Mean Squared Error
    'rmse',                     # Root Mean Squared Error
    'mae',                      # Mean Absolute Error
    'r2_score',                 # R2 Score (Coefficient of determination)
    'explained_variance',       # Explained Variance Score
    'max_error',                # Maximum Error
])

def train_random_forest(X_train, y_train, X_test, y_test):
    try:
        def objective(trial):
            # Suggest hyperparameters for RandomForest
            n_estimators = trial.suggest_int('n_estimators', 10, 100)  # Number of trees
            max_depth = trial.suggest_int('max_depth', 3, 20)  # Max depth of trees
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)  # Minimum samples required to split a node
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)  # Minimum samples required at leaf node
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])  # Feature selection for splits

            # Create and train the RandomForest model with the suggested hyperparameters
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)  # 3-fold cross-validation
            return score.mean()  # Return mean cross-validation score

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')  # Maximizing accuracy
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the top 10 trials based on accuracy
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        # Access the global DataFrame
        global global_model_results

        for trial in best_trials:
            params = trial.params
            # Train the model with the best hyperparameters
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)

            # Save the model to file
            model_filename = f"best_random_forest_model_trial_{trial.number}.pkl"

            # Append details to the global DataFrame
            global_model_results = pd.concat([
                global_model_results,
                pd.DataFrame([{
                    "model_name": f"RandomForest_trial_{trial.number}",
                    "algorithm": "RandomForest",
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_rep,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 RandomForest Models stored successfully!")

    except Exception as e:
        print(f"Error in RandomForest training: {str(e)}")

def train_decision_tree(X_train, y_train, X_test, y_test):
    try:
        def objective(trial):
            # Suggest hyperparameters for Decision Tree
            max_depth = trial.suggest_int('max_depth', 1, 50)  # Maximum depth of the tree
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)  # Minimum samples required to split
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)  # Minimum samples at a leaf node
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])  # Split quality criterion

            # Create and train the Decision Tree model with the suggested hyperparameters
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42
            )

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)  # 3-fold cross-validation
            return score.mean()  # Return mean cross-validation score

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')  # Maximizing accuracy
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the best hyperparameters and top 10 trials
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        # Add results to the global DataFrame
        global global_model_results  # Access the global DataFrame

        for trial in best_trials:
            params = trial.params
            # Train the model with the best hyperparameters
            model = DecisionTreeClassifier(**params, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)

            # Save the model to file
            model_filename = f"best_decision_tree_model_trial_{trial.number}.pkl"
            # joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results = pd.concat([
                global_model_results,
                pd.DataFrame([{
                    "model_name": f"DecisionTree_trial_{trial.number}",
                    "algorithm": "DecisionTree",
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_rep,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 Decision Tree Models stored successfully!")

    except Exception as e:
        print(f"Error in Decision Tree training: {str(e)}")

def train_knn(X_train, y_train, X_test, y_test):
    try:
        def objective(trial):
            # Suggest hyperparameters for KNN
            n_neighbors = trial.suggest_int('n_neighbors', 1, 50)  # Number of neighbors
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])  # Weight function
            p = trial.suggest_int('p', 1, 2)  # Distance metric (1: Manhattan, 2: Euclidean)

            # Create and train the KNN model with the suggested hyperparameters
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)  # 3-fold cross-validation
            return score.mean()  # Return mean cross-validation score

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')  # Maximizing accuracy
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the best hyperparameters and top 10 trials
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        # Access the global DataFrame
        global global_model_results

        for trial in best_trials:
            params = trial.params
            # Train the model with the best hyperparameters
            model = KNeighborsClassifier(**params)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)

            # Save the model to file
            model_filename = f"best_knn_model_trial_{trial.number}.pkl"
            # joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results = pd.concat([
                global_model_results,
                pd.DataFrame([{
                    "model_name": f"KNN_trial_{trial.number}",
                    "algorithm": "KNN",
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_rep,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 KNN Models stored successfully!")

    except Exception as e:
        print(f"Error in KNN training: {str(e)}")

def train_logistic_regression(X_train, y_train, X_test, y_test):
    try:
        def objective(trial):
            # Suggest hyperparameters for Logistic Regression
            C = trial.suggest_loguniform('C', 1e-3, 1e3)  # Regularization strength
            solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'sag', 'saga'])  # Solver type
            max_iter = trial.suggest_int('max_iter', 100, 1000, step=100)  # Maximum iterations

            # Create and train the Logistic Regression model with suggested hyperparameters
            model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=3, n_jobs=-1)  # 3-fold CV
            return score.mean()  # Return mean cross-validation score

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')  # Maximizing accuracy
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the top 10 trials based on accuracy
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        # Update the global DataFrame
        global global_model_results

        for trial in best_trials:
            params = trial.params
            # Train the model with the best hyperparameters
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)

            # Save the model to file
            model_filename = f"best_logistic_model_trial_{trial.number}.pkl"
            # joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results = pd.concat([
                global_model_results,
                pd.DataFrame([{
                    "model_name": f"LogisticRegression_trial_{trial.number}",
                    "algorithm": "Logistic Regression",
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_rep,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 Logistic Regression Models stored successfully!")

    except Exception as e:
        print(f"Error in Logistic Regression training: {str(e)}")
def train_svc(X_train, y_train, X_test, y_test):
    try:
        def objective(trial):
            # Suggest hyperparameters for SVC
            C = trial.suggest_loguniform('C', 1e-3, 1e3)  # Regularization parameter
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])  # Kernel type
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])  # Kernel coefficient

            # Create and train the SVC model with the suggested hyperparameters
            model = SVC(C=C, kernel=kernel, gamma=gamma)

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)  # 3-fold cross-validation
            return score.mean()  # Return mean cross-validation score

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')  # Maximizing accuracy
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the top 10 trials based on accuracy
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        # Access the global DataFrame
        global global_model_results

        for trial in best_trials:
            params = trial.params
            # Train the model with the best hyperparameters
            model = SVC(**params)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)

            # Save the model to file
            model_filename = f"best_svc_model_trial_{trial.number}.pkl"
            # joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results = pd.concat([
                global_model_results,
                pd.DataFrame([{
                    "model_name": f"SVC_trial_{trial.number}",
                    "algorithm": "SVC",
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_rep,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 SVC Models stored successfully!")

    except Exception as e:
        print(f"Error in SVC training: {str(e)}")


def train_svr(X_train,X_test, y_train, y_test):
    try:
        # Define the objective function for Optuna
        def objective(trial):
            # Suggest hyperparameters for SVR
            C = trial.suggest_float('C', 0.1, 1000.0, log=True)  # Regularization parameter
            epsilon = trial.suggest_float('epsilon', 0.01, 1.0, log=True)  # Epsilon-insensitive loss
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])  # Kernel type

            # Create and train the SVR model with the suggested hyperparameters
            model = SVR(C=C, epsilon=epsilon, kernel=kernel)

            # Perform cross-validation to evaluate the model
            score = cross_val_score(
                model, X_train, y_train, n_jobs=-1, cv=3, scoring='neg_mean_squared_error'
            )
            return -score.mean()  # Return mean negative MSE (minimizing)

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='minimize')  # Minimizing MSE
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the top 10 trials based on MSE
        best_trials = sorted(study.trials, key=lambda t: t.value)[:10]

        # Access the global DataFrame
        global global_model_results2

        for trial in best_trials:
            params = trial.params

            # Train the model with the best hyperparameters
            model = SVR(**params)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
            r2 = r2_score(y_test, y_pred)  # R2 Score
            explained_var = explained_variance_score(y_test, y_pred)  # Explained Variance
            max_error_val = max_error(y_test, y_pred)  # Max Error

            # Save the model to file
            model_filename = f"best_svr_model_trial_{trial.number}.pkl"
            # joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results2 = pd.concat([
                global_model_results2,
                pd.DataFrame([{
                    "model_name": f"SVR_trial_{trial.number}",
                    "algorithm": "SVR",
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2_score": r2,
                    "explained_variance": explained_var,
                    "max_error": max_error_val,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 SVR Models stored successfully!")

    except Exception as e:
        print(f"Error in SVR training: {str(e)}")

def train_linear_regression(X_train,X_test, y_train, y_test):
    try:
        # Define the objective function for Optuna
        def objective(trial):
            # Suggest hyperparameters for Ridge Regression (as LinearRegression has no hyperparameters)
            alpha = trial.suggest_float('alpha', 0.1, 100.0, log=True)  # Regularization strength for Ridge Regression

            # Create and train the Ridge model with the suggested hyperparameters
            model = Ridge(alpha=alpha)

            # Perform cross-validation to evaluate the model
            score = cross_val_score(
                model, X_train, y_train, n_jobs=-1, cv=3, scoring='neg_mean_squared_error'
            )
            return -score.mean()  # Return mean negative MSE (minimizing)

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='minimize')  # Minimizing MSE
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the top 10 trials based on MSE
        best_trials = sorted(study.trials, key=lambda t: t.value)[:10]

        # Access the global DataFrame
        global global_model_results2

        for trial in best_trials:
            params = trial.params

            # Train the model with the best hyperparameters
            model = Ridge(**params)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
            r2 = r2_score(y_test, y_pred)  # R2 Score
            explained_var = explained_variance_score(y_test, y_pred)  # Explained Variance
            max_error_val = max_error(y_test, y_pred)  # Max Error

            # Save the model to file
            model_filename = f"best_linear_regression_model_trial_{trial.number}.pkl"
            # joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results2 = pd.concat([
                global_model_results2,
                pd.DataFrame([{
                    "model_name": f"LinearRegression_trial_{trial.number}",
                    "algorithm": "Linear Regression",
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2_score": r2,
                    "explained_variance": explained_var,
                    "max_error": max_error_val,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 Linear Regression Models stored successfully!")

    except Exception as e:
        print(f"Error in Linear Regression training: {str(e)}")


#
# if __name__ == "__main__":
#     # Train all models
#     train_knn()
#     train_svc()
#     train_logistic_regression()
#     train_decision_tree()
#     train_random_forest()
#
#     # Dump selected models
#     dump_models()

    # Print the global results
    # print(global_model_results.head())