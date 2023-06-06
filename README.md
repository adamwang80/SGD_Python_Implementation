# Stochastic Gradient Descent Implementation and Application

**Authors:** Hangxiao Zhu, Adam Wang

## Overview

This project aims to implement the Stochastic Gradient Descent (SGD) algorithm and apply it to a generated dataset, with the goal of finding the weights that minimize true risk. The effects of different hyperparameters, such as variance and training size, are also examined, noting their impact on logistic loss and classification error.

## Algorithm Implementation

The SGD algorithm is implemented as follows:

### Data Generation and Euclidean Projection

Two functions, `generate_data` and `euclidean_projection`, are used for data generation and computation. 

The `generate_data` function initializes the data as a 4x30 numpy array filled with initial values, and initializes the label as a 1x30 numpy array filled with initial values. A random value between 0 and 1 is assigned to each data point. Depending on this value, the label is set and a Gaussian vector `u` is generated. The vector `u` is then projected to `X` to get the desired data.

The `euclidean_projection` function computes the Euclidean projection of a point to the parameter set `C`. Since the feature space `X` and the parameter space `C` are both unit balls centered around the origin, the projection process is essentially the process of computing the unit vector. The `numpy.linalg.norm` function is used to compute the distance of the point to the origin, and the point vector is then divided by this distance to get the projected vector.

## Experiments

Experiments are conducted with different combinations of parameters for SGD, such as learning rate and training size. During these experiments, data with different `σ` values and training sizes are generated, and the results are computed. 

Two arrays are used to store excess risks and mean errors for each scenario, facilitating data visualization. 

## Analysis of Lipschitzness and Boundedness properties

For the logistic loss function `l(w,(x,y))`, it is convex and `||x||`-Lipschitz, therefore `ρ = 5`. The diameter of the parameter set `C`, which is a unit ball, is `2`, therefore `M = 2`.

## Results

The results are presented in various plots showing the standard deviation of loss and error for different `σ` values.

## Conclusion

The test results generally align with theoretical predictions. The SGD algorithm progressively reduces the error function value as parameters are iteratively updated. The rate of error function reduction is faster for a dataset with a smaller standard deviation (`σ = 0.1`) compared to one with a larger standard deviation (`σ = 0.35`). This suggests that the SGD algorithm can more easily learn from a dataset with a smaller standard deviation. 

When the dataset is too scattered, or has a larger standard deviation, more iterations and self-correction time are needed for the algorithm to find the optimal solution. This is due to the updated weight values determining the direction and magnitude of the algorithm's next move, and a more scattered dataset means these weights will be more variable.

## Future Work

Further improvements and extensions to this project could include:

1. Exploring different initialization strategies for the SGD algorithm.
2. Examining the performance of SGD under different loss functions beyond logistic loss.
3. Running the same experiments with larger datasets to verify the scalability of the SGD algorithm.
4. Evaluating the impact of different learning rates on the performance of SGD.
