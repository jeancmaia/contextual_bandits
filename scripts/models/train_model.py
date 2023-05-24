import logging
import pandas as pd
import numpy as np

from joblib import dump

from agent_model.estimator import Agent

DESTINATION = "assets/models/model.joblib"
BATCH_SIZE = 15
DAY = "day"
RAW_PRICE = "raw_price"
P1_BINOMIAL = "p1_binomial"
P2_BINOMIAL = "p2_binomial"
P3_BINOMIAL = "p3_binomial"
P4_BINOMIAL = "p4_binomial"


def get_data_processed(processed_dataset="assets/processed/data_event.parquet"):
    return pd.read_parquet(processed_dataset)


def create_batch_prediction(events_dataset, agent):
    logger = logging.getLogger(__name__)
    logger.info("batch size " + str(BATCH_SIZE) + " for training")

    for i in range(int(np.floor(events_dataset.shape[0] / BATCH_SIZE))):
        batch_st = (i + 1) * BATCH_SIZE
        batch_end = (i + 2) * BATCH_SIZE
        batch_end = np.min([batch_end, events_dataset.shape[0]])

        X_batch = events_dataset.iloc[batch_st:batch_end][[DAY, RAW_PRICE]]
        y_batch = events_dataset.iloc[batch_st:batch_end][
            [P1_BINOMIAL, P2_BINOMIAL, P3_BINOMIAL, P4_BINOMIAL]
        ].values

        actions_this_batch = agent.predict(X_batch).astype("uint8")
        rewards_batch = y_batch[np.arange(y_batch.shape[0]), actions_this_batch]
        if i % 100 == 0:
            logger.info("batch number : " + str(i))
            logger.info(
                "total reward for batch " + str(i) + " is: " + str(rewards_batch.sum())
            )
        agent.partial_fit(X_batch, actions_this_batch, rewards_batch)
    return agent


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making dataset with conversion events and its covariates.")

    events_dataset = get_data_processed()
    logger.info("dataset extracted.")
    logger.info("columns: " + str(events_dataset.columns))

    agent = create_batch_prediction(events_dataset, Agent())

    dump(agent, DESTINATION)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
