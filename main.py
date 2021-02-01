import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

tfd = tfp.distributions
#The first day in our sequence has an 30% chance of being cold
initial_distribution = tfd.Categorical(probs=[0.3, 0.7])
#A cold day has a 50% chance of being followed by a hot day.
#A hot day has a 20% chance of being followed by a cold day
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])
#On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 15 and 10 on a hot day.
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()


with tf.compat.v1.Session() as sess:
  print(mean.numpy())