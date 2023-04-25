
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib 

# Get data: Preprocess or 
with open('preprocessed_data_en_all_levels.pkl', 'rb') as file:
    dataset_dict = pickle.load(file)
X_train, X_test, Y_train, Y_test = dataset_dict["X_train"], dataset_dict["X_test"], dataset_dict["Y_train"], dataset_dict["Y_test"]

# Define logistic regression model
base_lr = LogisticRegression(solver='sag', random_state = 42, max_iter=1000, penalty = 'l2')
# Define ClassifierChain for multi-label task 
clf = ClassifierChain(base_lr, verbose=True)

# Set up pipeline with vectorizer and classifier
pipeline = Pipeline([
    ('clf', clf)
])

# Set up hyperparameter grid for tuning
param_grid = {
    'clf__base_estimator__C': [0.001, 0.01, 0.1, 1, 10],
}

# Set up grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1_micro')

# grid_search.fit(X_train, Y_train)
clf.fit(X_train, Y_train)

# Save the best model locally
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.joblib')

predictions = best_model.predict(X_test)
joblib.dump(predictions, 'predictions.joblib')

