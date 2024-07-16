import os
from omegaconf import OmegaConf
from logger import Logger
from model_utils import merge_conf
from trainer import MusCALLTrainer
import mlflow
import mlflow.pytorch
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def main():

    base_conf_path = 'configs/base_config.yaml'
    dataset_conf_path = 'configs/dataset.yaml'
    model_conf_path = 'configs/model.yaml'
    muscall_config = merge_conf(base_conf_path=base_conf_path, dataset_conf_path=dataset_conf_path, model_conf_path=model_conf_path)

    logger = Logger(muscall_config)

    trainer = MusCALLTrainer(muscall_config, logger)

    experiment_path = os.path.join(muscall_config.env.experiments_dir, muscall_config.env.experiment_id)

    # Set the experiment name
    mlflow.set_experiment(experiment_path)

    # Start an MLflow run
    with mlflow.start_run():
        trainer.train()


if __name__ == "__main__":
    main()