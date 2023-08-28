import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.multioutput import ClassifierChain

import wandb
from config.data_args import DataArgs
from src.preprocessing.get_preprocessed_data import get_preprocessed_data

wandb.login()
model_name = 'logistic_regression'

for random_seed in [42,43,44,45,46]:
    print('----------------------------------------Random seed:',{random_seed})

    name_run = f'{model_name}_{random_seed}'
    wandb_project = f'multi-label-{model_name}'
    wandb_kwargs = {"name": name_run}
    
    # Initialize wandb
    run = wandb.init(project=wandb_project, name = name_run)

    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_preprocessed_data(model_name, overwrite_data= False, random_seed= random_seed)      

    do_hypertuning = True

    if do_hypertuning:
        # Set up hyperparameter grid for tuning 
        param_grid = {
            'base_estimator__C': [0.1, 1],
            'base_estimator__penalty': ['l1', 'l2']
        }
        print('Hypertuning starts.')
        best_score = 0 
        for param in ParameterGrid(param_grid):
            print(param)

            # Define logistic regression model
            base_lr = LogisticRegression(solver='liblinear', random_state = random_seed, max_iter=1000)

            # Define ClassifierChain for multi-label task 
            clf = ClassifierChain(base_lr, verbose = False)

            clf.set_params(**param)
            clf.fit(X_train.toarray(),Y_train)

            predictions_val = clf.predict(X_val.toarray()) 

            # log evaluation metrics
            f1_score_avg = f1_score(Y_val, predictions_val, average='samples', zero_division=0)
            label_ranking_score = label_ranking_average_precision_score(Y_val, predictions_val)
            f1_score_macro = f1_score(Y_val, predictions_val, average='macro', zero_division=0)
            f1_score_micro = f1_score(Y_val, predictions_val, average='micro', zero_division=0)
            print(f'eval_LRAP {label_ranking_score}, eval_f1_score_avg: {f1_score_avg}, eval_f1_score_macro: {f1_score_macro}, eval_f1_score_micro: {f1_score_micro}')
        
            # save if best
            if f1_score_avg > best_score:
                best_score = f1_score_avg
                best_param = param
                best_model = clf

        predictions = best_model.predict(X_test.toarray()) 
    else:
        # Define logistic regression model
        base_lr = LogisticRegression(solver='liblinear', random_state = random_seed, max_iter=1000, penalty='l1', C =0.1)

        # Define ClassifierChain for multi-label task 
        clf = ClassifierChain(base_lr, verbose = False)
        
        clf.fit(X_train.toarray(),Y_train)

        predictions = best_model.predict(X_test.toarray()) 

    if not os.path.exists('outputs/logistic_regression'):
        os.makedirs('outputs/logistic_regression')
    joblib.dump(predictions, f'outputs/logistic_regression/predictions_{random_seed}.joblib')
    
    args = DataArgs()
    # log evaluation metrics
    f1_score_avg = f1_score(Y_test, predictions, average='samples', zero_division=0)
    label_ranking_score = label_ranking_average_precision_score(Y_test, predictions)
    f1_score_macro = f1_score(Y_test, predictions, average='macro', zero_division=0)
    f1_score_micro = f1_score(Y_test, predictions, average='micro', zero_division=0)
    f1_score_avg_1 = f1_score(Y_test, predictions, labels = range(6), average='samples', zero_division=0)
    f1_score_avg_3 = f1_score(Y_test, predictions, labels = range(6,42), average='samples', zero_division=0)
    f1_score_avg_5 = f1_score(Y_test, predictions, labels = range(42,260), average='samples', zero_division=0)

    run.log({"test_LRAP": label_ranking_score, 
             "test_f1_score_avg": f1_score_avg, 
             "test_f1_score_macro": f1_score_macro, 
             "test_f1_score_micro": f1_score_micro,
             "test_f1_score_avg_1": f1_score_avg_1,
             "test_f1_score_avg_3": f1_score_avg_3,
             "test_f1_score_avg_5": f1_score_avg_5,
             })

    # log the data as an artifact
    data_artifact = wandb.Artifact("data", "dataset")
    data_artifact.add_file(f"data/preprocessed/logistic_regression_en_all_levels_val_{random_seed}/preprocessed_data.pkl")
    run.log_artifact(data_artifact)
    

    # log the data config as an artifact
    config_artifact = wandb.Artifact("config", type="config")
    config_artifact.add_file(f"data/preprocessed/logistic_regression_en_all_levels_val_{random_seed}/data_args.json")
    run.log_artifact(config_artifact)

    # log the model as an artifact
     # Save the best model locally
    joblib.dump(best_model, f'outputs/logistic_regression/best_model_{random_seed}.joblib')

     # Save the best model to W&B
    best_model_artifact = wandb.Artifact(f"model_{random_seed}", type='model')
    best_model_artifact.add_file(f'outputs/logistic_regression/best_model_{random_seed}.joblib')
    run.log_artifact(best_model_artifact)

    # finish logging the data logging run
    run.finish()

