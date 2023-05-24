import logging
import pandas as pd
import numpy as np

from pathlib import Path

BASE_CONVERSION = 0.25
DAYS_OF_EVALUATION = 365
WEEK = 7
TWO_MONTHS = 60
YEAR = 365
EVENTS_PER_DAY = 500
DATA_SOURCE = "assets/processed"
FILE_DATASOURCE = "data_event.parquet"
POSITIVE = "positive"
INVERSE = "inverse"


STR_BASE_CONVERSION = "base_conversion"
STR_TWO_MONTHS_SEASONALITY = "two_months_seasonality"
STR_TWO_MONTHS_SEASONALITY__SIGNAL = "two_months_seasonality__signal"
STR_WEEK_SEASONALITY = "week_seasonality"
STR_WEEK_SEASONALITY__SIGNAL = "week_seasonality__signal"
STR_PRICE_COEFFICIENT = "price_coefficient"
STR_BETA_NOISE = "beta_noise"
STR_BETA__ALPHA = "beta__alpha"
STR_BETA__BETA = "beta__beta"
STR_RAW_PRICE = "raw_price"
STR_DAY = "day"

STR_SUFFIX_P = "_p"
STR_SUFFIX_BINOMIAL = "_binomial"


def generate_week_seasonality(days_of_evaluation=DAYS_OF_EVALUATION):
    """
    It produces week seasonal components for horizon of size `days_of_evaluation`.
    """
    weekly_periods = np.array(list(range(days_of_evaluation))) % YEAR / WEEK
    n_order = 1
    return (np.sin(np.pi * weekly_periods * n_order) + 1) / 2


def generate_two_months_seasonality(days_of_evaluation=DAYS_OF_EVALUATION):
    """
    It produces two months seasonal components for horizon of size `days_of_evaluation`.
    """
    two_month_periods = np.array(list(range(days_of_evaluation))) % YEAR / TWO_MONTHS
    n_order = 1
    return (np.sin(np.pi * two_month_periods * n_order) + 1) / 2


ARMS = {
    "p1": {
        STR_BASE_CONVERSION: BASE_CONVERSION,
        STR_TWO_MONTHS_SEASONALITY: 0.1,
        STR_TWO_MONTHS_SEASONALITY__SIGNAL: POSITIVE,
        STR_WEEK_SEASONALITY: 0,
        STR_WEEK_SEASONALITY__SIGNAL: INVERSE,
        STR_PRICE_COEFFICIENT: 0,
        STR_BETA_NOISE: 0.2,
        STR_BETA__ALPHA: 1,
        STR_BETA__BETA: 1.2,
    },
    "p2": {
        STR_BASE_CONVERSION: BASE_CONVERSION,
        STR_TWO_MONTHS_SEASONALITY: 0.08,
        STR_TWO_MONTHS_SEASONALITY__SIGNAL: INVERSE,
        STR_WEEK_SEASONALITY: 0,
        STR_WEEK_SEASONALITY__SIGNAL: INVERSE,
        STR_PRICE_COEFFICIENT: 0,
        STR_BETA_NOISE: 0.25,
        STR_BETA__ALPHA: 1,
        STR_BETA__BETA: 1.2,
    },
    "p3": {
        STR_BASE_CONVERSION: BASE_CONVERSION,
        STR_TWO_MONTHS_SEASONALITY: 0,
        STR_TWO_MONTHS_SEASONALITY__SIGNAL: INVERSE,
        STR_WEEK_SEASONALITY: 0.05,
        STR_WEEK_SEASONALITY__SIGNAL: INVERSE,
        STR_PRICE_COEFFICIENT: 0.2,
        STR_BETA_NOISE: 0.1,
        STR_BETA__ALPHA: 1,
        STR_BETA__BETA: 1.2,
    },
    "p4": {
        STR_BASE_CONVERSION: BASE_CONVERSION,
        STR_TWO_MONTHS_SEASONALITY: 0,
        STR_TWO_MONTHS_SEASONALITY__SIGNAL: INVERSE,
        STR_WEEK_SEASONALITY: 0.1,
        STR_WEEK_SEASONALITY__SIGNAL: INVERSE,
        STR_PRICE_COEFFICIENT: 0.15,
        STR_BETA_NOISE: 0.05,
        STR_BETA__ALPHA: 1,
        STR_BETA__BETA: 1.2,
    },
}


def generate_p_binomial(
    arm_conf, day, raw_price, week_seasonality, two_month_seasonality
):
    week_seasonal_component = (
        (1 - week_seasonality[day])
        if arm_conf.get(STR_WEEK_SEASONALITY__SIGNAL) == INVERSE
        else week_seasonality[day]
    )

    two_months_seasonal_component = (
        (1 - two_month_seasonality[day])
        if arm_conf.get(STR_TWO_MONTHS_SEASONALITY__SIGNAL) == INVERSE
        else two_month_seasonality[day]
    )

    return (
        BASE_CONVERSION
        + (arm_conf.get(STR_TWO_MONTHS_SEASONALITY) * two_months_seasonal_component)
        + (arm_conf.get(STR_WEEK_SEASONALITY) * week_seasonal_component)
        + arm_conf.get(STR_PRICE_COEFFICIENT) * raw_price
        + np.random.beta(
            arm_conf.get(STR_BETA__ALPHA), arm_conf.get(STR_BETA__BETA), size=1
        )[0]
        * arm_conf.get(STR_BETA_NOISE)
    )


def main(output_filepath, week_seasonality, two_month_seasonality):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making dataset with conversion events and its covariates.")
    logger.info("number of days for evaluation: " + str(DAYS_OF_EVALUATION))

    beta_alpha = 1
    beta_beta = 1.2

    simulation_data = []
    for day in range(DAYS_OF_EVALUATION):
        for _ in range(
            np.random.random_integers(EVENTS_PER_DAY, EVENTS_PER_DAY + 20, 1)[0]
        ):
            raw_price = np.random.beta(beta_alpha, beta_beta, size=1)[0]

            events = dict()
            for arm_position in ARMS.keys():
                p = generate_p_binomial(
                    ARMS[arm_position],
                    day,
                    raw_price,
                    week_seasonality,
                    two_month_seasonality,
                )
                events[arm_position + STR_SUFFIX_P] = p

                binomial_event = np.random.binomial(
                    1, events[arm_position + STR_SUFFIX_P], 1
                )[0]
                events[arm_position + STR_SUFFIX_BINOMIAL] = binomial_event
                events[STR_DAY] = day
                events[STR_RAW_PRICE] = raw_price

            simulation_data.append(events)

    data_simulation = pd.DataFrame(simulation_data)
    logger.info("persisting dataset with size: " + str(data_simulation.shape))
    data_simulation.to_parquet(output_filepath)
    logger.info("dataset persisted successfully")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    two_month_seasonality = generate_two_months_seasonality()
    week_seasonality = generate_week_seasonality()

    destination = Path(DATA_SOURCE).joinpath(FILE_DATASOURCE)

    main(destination, week_seasonality, two_month_seasonality)
