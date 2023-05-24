contextual_bandits
==============================


This project evaluates contextual bandits on a simulated dataset, suggesting four possible discounts and their returns.


Contextual Bandits are also known as Bandits with covariates, which can improve Upper Confidence Bound Bandits to non-stochastic outcomes. We built synthetic data, combining different seasonalities and noises to create a dynamic system to behold whether CB can find optimal policies. The Agent - the model that pivots to the best arm, had great behavior and reached heydays conversion overall.

Check out >reports/ContextualBandits.md to deep details about the model.

The Makefile stores all management flows, from dataset generation, encompassing model training, to model API.