import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):

    mode = cfg.run_mode.name
    print("Run mode:", mode)
    if mode == "preprocessing":
        from src import create_dataset
        create_dataset.main(cfg)


    elif mode == "training":
        from . import run_training
        run_training.main(cfg)
    else:
        print("Run mode not specified. Typo?")


if __name__ == "__main__":
    run()