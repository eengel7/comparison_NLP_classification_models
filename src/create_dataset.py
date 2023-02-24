from src.preprocessing.preprocessor import LogRegPreprocessor 

def main(cfg):

    # Call preprocessor depending on the model
    if cfg.model.name == "logistic_regression":  
            preprocessor = LogRegPreprocessor(cfg)
        # else:
        #     preprocessor = FastTextPreprocessor(cfg)

    preprocessor.preprocess_and_store()