import hydra


def get_data_dir(cfg):
    emb_name = cfg.embedding.name
    data_settings = cfg.dev_settings if not split_settings else split_settings
    preprocessed_path = hydra.utils.to_absolute_path(cfg.paths.preprocessed_path)

    if emb_name != "transformer":
        print(data_settings)
        dest_dir = os.path.join(
            preprocessed_path,
            f"emb-{emb_name}"
            f"_tuned-{cfg.pre_processing.tune}{cfg.pre_processing.epochs}"
            f"_uncertainremoved-{cfg.pre_processing.remove_uncertain}"
            f"_annotated-{data_settings.annotation}"
            f"_tfidf-{cfg.pre_processing.use_tfidf}"
            f"_avgemb-{cfg.pre_processing.mean}"
            f"_balanced-{cfg.balance_on_majority}"
            f"_gendered-{data_settings.augment}",
        )
    else:
        dest_dir = os.path.join(
            preprocessed_path,
            f"emb-{emb_name}"
            f"_uncertainremoved-{cfg.pre_processing.remove_uncertain}"
            f"_annotated-{data_settings.annotation}"
            f"_balanced-{cfg.balance_on_majority}"
            f"_gendered-{data_settings.augment}",
        )
    os.makedirs(dest_dir, exist_ok=True)
    return dest_dir
