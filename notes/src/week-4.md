---
title: "Machine Learning: Representing Neural Networks"
author: "Sidharth Baskaran"
date: "June 2021"
---

# Non-linear hypotheses

* Computer vision - section matrix of pixel intensities correspond to an image
  * Denote + and – for affirming if an image fits a classification
  * Would need nonlinear hypothesis
  * Feature vector $x$ - pixel intensities in a column vector
* Example
  * Assume $50\times 50$ pixel images - 2500 pixels
    * $n=2500$ features
  * Quadratic features would mean ~ 3 mil features
    * $\text{number of features}=50^2 + C(50^2,2)$ as need all possible ways of 2 terms from features in addition to number of features present

# Neural Network Model

* Neuron structure
  * Dendrite - input wires
  * Computation in nucleus
  * Axon - output wires
* Neuron model - logistic unit
  * Input wires from features $x$ through computation to output $h_\theta(x)$
  * Sigmoid activation function $g(z)=\frac{1}{1+e^{-z}}$
  * Parameters $\theta$ are same as weights
* Layers of neural networks
  * Layer 1 of features/inputs
  * Layer 2 of bias units - is *hidden* as is not an output
  * Layer 3 is the output
  
$$
\left[x_{0} x_{1} x_{2} x_{3}\right] \rightarrow\left[a_{1}^{(2)} a_{2}^{(2)} a_{3}^{(2)}\right] \rightarrow h_{\theta}(x)
$$

Node values are

$$
\begin{array}{r}
a_{1}^{(2)}=g\left(\Theta_{10}^{(1)} x_{0}+\Theta_{11}^{(1)} x_{1}+\Theta_{12}^{(1)} x_{2}+\Theta_{13}^{(1)} x_{3}\right) \\
a_{2}^{(2)}=g\left(\Theta_{20}^{(1)} x_{0}+\Theta_{21}^{(1)} x_{1}+\Theta_{22}^{(1)} x_{2}+\Theta_{23}^{(1)} x_{3}\right) \\
a_{3}^{(2)}=g\left(\Theta_{30}^{(1)} x_{0}+\Theta_{31}^{(1)} x_{1}+\Theta_{32}^{(1)} x_{2}+\Theta_{33}^{(1)} x_{3}\right) \\
h_{\Theta}(x)=a_{1}^{(3)}=g\left(\Theta_{10}^{(2)} a_{0}^{(2)}+\Theta_{11}^{(2)} a_{1}^{(2)}+\Theta_{12}^{(2)} a_{2}^{(2)}+\Theta_{13}^{(2)} a_{3}^{(2)}\right)
\end{array}
$$

**Arguments are $z_c^{(k)}$ where $c$ is which element of the layer and $k$ is the layer.**

* Notation
  * $a_i^{(j)}$ is activation of unit $i$ in layer $j$
  * $\Theta^{(j)}$ is matrix of weights controlling function mapping from layer $j$ to layer $j+1$
  * If network has $s_j$ units in layer $j$, $s_{j+1}$ units in layer $j+1$, then $\Theta^{(j)}$ is of dimension $\boxed{s_{j+1}\times (s_j+1)}$
    * $x_0$ and $\Theta_0^{(j)}$ bias nodes are not shown in a NN diagram
* Vectorized
  * Arguments of $g$, $z_c^{(k)}=\theta^{(k)}x$
  * Can let $a^{(k)}=g(z^{(k)})=\Theta^{(k)}a^{(k)}$
  * $x=\left[\begin{array}{c}x_{0} \\x_{1} \\\cdots \\x_{n}\end{array}\right],\; z^{(j)}=\left[\begin{array}{c}z_{1}^{(j)} \\z_{2}^{(j)} \\\cdots \\z_{n}^{(j)}\end{array}\right]$
  * Thus $z^{(j)}=\Theta^{(j-1)}a^{(j-1)}$

# Multiclass Classification

* One-vs-all method extension
* Multiple output units for multiple classifications
* $h_\Theta(x)$ is a vector