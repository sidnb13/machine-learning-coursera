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

# Machine Learning System Design

* Supervised learning $\rightarrow$ $x$ = features of email, $y\in\{0,1\}$, can choose 100 words indicative of spam/not for features
* Can encode an email into a feature vector
  * In practice $\rightarrow$ take most frequently occurring $n$ words in training set
* Could develop features to reduce errors, e.g. misspelling detection
  * Equal consideration of all options $\rightarrow$ cannot tell which will work best

# Error Analysis

* Start with **simple** algorithm to test on CV data
* Plot learning curves to decide if more data, features, etc.
* Error analysis $\rightarrow$ manual examination of examples where errors occurred
  * Look for systematic error trend
  * Use evidence to guide decision-making not guesswork
* Numerical evaluation
  * Treating stem of word = to variants of word
  * Can use stemming software
  * Naturally $\rightarrow$ CV error $J_\mathrm{CV}(\theta)$ of algorithm with/without stemming and choose best options
    * Do not $J_\mathrm{train}(\theta)$ to allow for generalization

# Error metrics for skewed classes

* Precision/recall
  * **Precision** $\rightarrow$ Of all predictions $y=1$, fraction that actually correspond to $y=1$
    * $\text{Precision}=\frac{\text{True pos.}}{\text{Predicted pos.}}=\frac{\text{True pos.}}{\text{True pos. + False pos.}}$
  * **Recall** $\rightarrow$ Of all actual cases $y=1$, what fraction was correctly detected by algorithm
    * $\text{Recall}=\frac{\text{True pos.}}{\text{Actual pos.}}=\frac{\text{True pos.}}{\text{True pos. + False neg.}}$
* Tradeoff of precision/recall
  * If trying to have high confidence $\rightarrow$ high precision, low recall
  * If trying to minimize false negatives $\rightarrow$ high recall, low precision
* In general $\rightarrow$ predict 1 if $h_\theta(x)\geq \text{threshold}$
* $F_1$ score $\rightarrow$ comparing precision/recall numbers
  * Can calculate average $\frac{P+R}{2}$ but susceptible to high or low recall/precision weighted
  * $F_1$ or $F$-score better $\rightarrow$ $2\frac{PR}{P+R}$
    * Gives more weight to $\mathrm{min}(P,R)$

# Data for Machine Learning

* Large data rationale
  * Assume feature $x\in \mathbb{R}^{n+1}$ has enough informaiton to predict $y$ accurately
  * Can ask if given input $x$, can human expert confidently predict $y$
  * A learning algorithm with many parameters or NN with many hidden layers
    * $J_\mathrm{train}(\theta)$ very small
  * Very large training set $\rightarrow$ unlikely to overfit
    * $J_\mathrm{train}(\theta) \approx J_\mathrm{test}(\theta)$