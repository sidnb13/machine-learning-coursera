---
title: "Advice for Applying Machine Learning"
author: Sidharth Baskaran
date: June 2021
---

# Improving algorithm

* If there are large errors in predictions
  * More training examples
  * Smaller set of features
  * Additional features
  * Add polynomial features (i.e. $x_1x_2, x_1^2,x_2^2$)
  * Decrease or increase $\lambda$
* Machine learning diagnostic
  * Test run to understand performance of algorithm

# Evaluating a Hypothesis

* Low training error does not mean good $h_\theta(x)$
* Hard to plot hypothesis as $n\to \text{large}$
* Split data into 2 parts $\rightarrow$ test and training
* Training/testing procedure for linear regression
  * Leanr parameter $\theta$ from training data (minimize training error $J(\theta$)
  * Compute test error
    * $J_\mathrm{test}(\theta)=\frac{1}{2m_\mathrm{test}}\sum_{i=1}^{m_\mathrm{test}}(h_\theta(x^{(i)}_\mathrm{test})-y^{(i)}_\mathrm{test})^2$
    * Change appropriately for logistic regression
  * Misclassificaiton error (0/1 misclassificaiton error)
    * Test error = $\frac{1}{m_\mathrm{test}}\sum_{i=1}^{m_\mathrm{test}}\text{err}(h_\theta(x^{(i)}_\mathrm{test}),y^{(i)}_\mathrm{test})$
        * Is proportion of misclassified test data

$$
\mathrm{err}(h_\theta(x),y)=
\begin{cases}
    {\color{red} 1}\text{ $y=0$ if $h_\theta(x)\geq 0.5$ \textbf{or} $y=1$ if $h_\theta(x)<0.5$}\\
    \text{{\color{red}0} otherwise}
\end{cases}
$$

# Model Selection and training/validation/test sets

* Error of parameters as measured to fit to a set of data < than actual generalization error
* Let there be a list of hypotheses of varying polynomial degree $d$ and $i\in \mathrm{range}(1,d)$
  * Calculate $J_\mathrm{test}(\theta^{(i)})$ in range
  * Lowest $J_\mathrm{test}$ means best model $\rightarrow$ but not fair estimate of generalization $\rightarrow$ optimistic choice
* Split data set into 3 parts
  * E.g. 60% training, 20% cross-validation (CV), 20% test
  * Determine $J_\mathrm{train},J_\mathrm{CV},J_\mathrm{test}$
  * In list of polynomial models - find lowest $J_\mathrm{CV}$ from CV set
  * Estimate generalization error from polynomial indexed $d$ with lowest CV error

# Bias vs. Variance

* Increase degree $d$ of polynomial $\rightarrow$ decreases training error of polynomial
  * Test error very similar
* Cross-validation error initially decreases then increases as overfitting begins to occur
* **High bias** problem $\rightarrow$ low $d$ and a high error
  * High $J_\mathrm{train}(\theta)$ and $J_\mathrm{CV}(\theta)\approx J_\mathrm{train}(\theta)$
* **High variance** problem $\rightarrow$ high $d$ and a high error
  * $J_\mathrm{train}(\theta)$ is low
  * $J_\mathrm{CV}(\theta) \gg J_\mathrm{train}(\theta)$

# Regularization and bias/variance

* High $\lambda$ means penalized parameters $\theta$ so high bias $\rightarrow$ underfit
* Intermediate $\lambda$ $\rightarrow$ optimal
* Small $\lambda$ $\rightarrow$ high variance and overfit
* Choosing regularization parameter $\lambda$ 
  * $J_\mathrm{train}(\theta),J_\mathrm{CV}(\theta),J_\mathrm{test}(\theta)$ all do not have regularization term but $J(\theta)$ does
  * Try a range of $\lambda$, e.g. in multiples of 2 and calculate the corresponding $J_\mathrm{CV}(\theta)$
    * Pick choice with lowest CV error
  * Then apply to $J_\mathrm{test}(\theta)$ to check for good generalization
* Can then plot $J_\mathrm{train}(\theta)$ and $J_\mathrm{CV}(\theta)$ as a function of $\lambda$
  * $J_\mathrm{train}(\theta)$ will increase
  * $J_\mathrm{CV}(\theta)$ will be upward parabolic

# Learning Curves

* Plot either training or CV error 
* A small training set size $m$ means virtually no error
  * As $m$ increwases, avg. training error of hypotheses increases
* CV error is high (low generalization) $\rightarrow$ small $m$
  * Tend to decrease with $m$
* More training data does not help case of high bias
* High variance problem
  * CV error will decrease with higher $m$
* As $m$ increases, $J_\mathrm{CV}(\theta)-J_\mathrm{train}(\theta)$ and both curves approach each other

## High bias

| Value of $m$ | $J_\mathrm{train}(\theta)$ | $J_\mathrm{CV}(\theta)$ |
| ------------ | -------------------------- | ----------------------- |
| Low | Low | High |
| High | High | Low |

## High variance

| Value of $m$ | $J_\mathrm{train}(\theta)$ | $J_\mathrm{CV}(\theta)$ |
| ------------ | -------------------------- | ----------------------- |
| Low | Low | High |
| High | Increases | Decreases |

# Debugging learning algorithm

* Choices
  * More training examples ($m$) $\rightarrow$ fixes high variance
  * Smaller sets of features $\rightarrow$ fixes high variance
  * Additional features $\rightarrow$ fixed high bias
  * Adding polynomial features $\rightarrow$ fixes high bias
  * Decreasing $\lambda$ $\rightarrow$ fixes high bias
  * Increasing $\lambda$ $\rightarrow$ fixed high variance
* Small neural network $\rightarrow$ prone to underfitting due to less parameters, computationally cheaper
* Large neural network $\rightarrow$ prone to overfitting and computationally expensive
  * Can address with $\lambda$