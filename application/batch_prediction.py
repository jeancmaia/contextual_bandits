import logging
import argparse
import pandas as pd
import numpy as np


from joblib import load

MODEL_LOC = "assets/models/model.joblib"
BATCH_LOC = "assets/batch_prediction/prediction.parquet"

mdl = load(MODEL_LOC)


def batch_prediction(dataset_path: str, features_list: list):
    """base method to roll batch prediction.

    Args:
        dataset_path (str): dataset location for batch prediction
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting the batch prediction step")

    data_for_prediction = pd.read_parquet(dataset_path)
    data_for_prediction = data_for_prediction[features_list]
    logger.info("Dataset shape for prediction " + str(data_for_prediction.shape))

    data_batch_prediction = np.concatenate(
        [mdl.predict(chunk) for chunk in np.array_split(data_for_prediction, 50)]
    )
    logger.info("Prediction batch ran out successfully")
    data_batch_prediction = pd.DataFrame(data_batch_prediction, columns=["prediction"])
    data_batch_prediction.to_parquet(BATCH_LOC)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="assets/processed/data_event.parquet")
    parser.add_argument("--features_list", default=["raw_price", "day"])
    args = parser.parse_args()
    return args.__dict__


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args = parse_arguments()

    batch_prediction(**args)
