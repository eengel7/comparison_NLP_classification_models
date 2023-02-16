import hydra
from omegaconf import DictConfig



@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):

    mode = cfg.run_mode
    print("Run mode:", mode)
    if mode == "data":
        from src import create_dataset
        create_dataset.main(cfg)
    # elif mode == "log_regression":
        # from src import run_classifier
        # run_classifier.run(cfg)
    else:
        print("Run mode not implemented. Typo?")


if __name__ == "__main__":
    print('Start')
    run()