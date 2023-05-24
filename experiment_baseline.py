
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
import joblib 
from src.preprocessing.get_preprocessed_data import get_preprocessed_data
import scipy.sparse as sp
from sklearn.metrics import make_scorer, f1_score, label_ranking_average_precision_score
import wandb 
wandb.login()
model_name = 'logistic_regression'


for random_seed in [42]:
    print('----------------------------------------Random seed:',{random_seed})

    name_run = f'{model_name}_{random_seed}'
    wandb_project = f'multi-label-{model_name}'
    wandb_kwargs = {"name": name_run}
    # Initialize wandb
    run = wandb.init(project=wandb_project, name = name_run)

    X_train, X_test, X_val, Y_train, Y_test, Y_val = get_preprocessed_data(model_name, overwrite_data= False, random_seed= random_seed)      
    print('Shape of X_train', X_train.shape)
    # Define logistic regression model
    base_lr = LogisticRegression(solver='liblinear', random_state = random_seed, max_iter=1000)
    # Define ClassifierChain for multi-label task 
    clf = ClassifierChain(base_lr, verbose=True)


    # Set up hyperparameter grid for tuning
    param_grid = {
        'base_estimator__C': [0.01],
        #'penalty': ['l1', 'l2','elasticnet']
    }

    scorer = make_scorer(f1_score, average='samples')
    # Set up grid search with cross-validation

    grid_search = GridSearchCV(clf, param_grid, cv=[(X_val.toarray(), Y_val)], scoring=scorer)
    print('Hypertuning starts.')
    grid_search.fit(X_train.toarray(), Y_train)

    # Save the best model locally
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'best_model.joblib')
    run.save('best_model.joblib')

    predictions = best_model.predict(X_test)
    joblib.dump(predictions, 'predictions.joblib')
    run.save('predictions.joblib')

    # Log to wandb

    # log the data as an artifact
    data_artifact = wandb.Artifact("data", "dataset")
    data_artifact.add_file(f"data/preprocessed/logistic_regression_en_all_levels_val_{random_seed}/preprocessed_data.pkl")
    run.log_artifact(data_artifact)
    

    # log the data config as an artifact
    config_artifact = wandb.Artifact("config", type="config")
    config_artifact.add_file(f"data/preprocessed/logistic_regression_en_all_levels_val_{random_seed}/data_args.json")
    run.log_artifact(config_artifact)

    # log evaluation metrics
    f1_score_avg = f1_score(Y_test, predictions, average='samples', zero_division=0)
    label_ranking_score = label_ranking_average_precision_score(Y_test, predictions)
    f1_score_avg = f1_score(Y_test, predictions, average='samples', zero_division=0) 
    f1_score_macro = f1_score(Y_test, predictions, average='macro', zero_division=0)
    f1_score_micro = f1_score(Y_test, predictions, average='micro', zero_division=0)
    run.log({"test_LRAP": label_ranking_score, "test_f1_score_avg": f1_score_avg, "test_f1_score_macro": f1_score_macro, "test_f1_score_micro": f1_score_micro})

     # finish logging the data logging run
    run.finish()

