from src.preprocessing.preprocessor import LogRegPreprocessor

def main(cfg):
    # get df with all annotations
    if cfg.run_mode == "data":
            preprocessor = LogRegPreprocessor(cfg)
        # else:
        #     preprocessor = FastTextPreprocessor(cfg)


    preprocessor.preprocess_and_store()