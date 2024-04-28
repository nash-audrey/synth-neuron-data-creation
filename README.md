- Someplace in the .py file header there is a place to include or exclude different files with beta distribution parameters. These can be changed for different purposes
- Weird sinkhorn results come from looking at the beta distribution templates between 5 and 6, where there are 50 linear interpolations between them. These distributions, the synthetic data generator to pull samples from them, and some results, are saved in the .zip folder. not all relevant information is included, I think.
- typically use the distribtuions for the 10 original templates, and pull enough samples from these to mimic neural behavior. difficult if you try to pull samples with too high of a rate bc there is a built-in delay to simulate a refraction period.

********

In the .zip folder are images for all of the given 10 original beta distributions in synth_dat_generator and the r code we used to create them
- Found distribution parameters by optimizing distribution to center 50% or 90% or 95% of weight in certain areas
- Also included uniform dist as one of the options

Overall, used beta distributions because they are defined only on [0,1]
