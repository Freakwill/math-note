# Bayes Classifier

[TOC]



## Principle

basic formula

Combinatorial bayes formula
$$
p(c|x)\sim \prod_ip(x_i|c)p(c)\sim \prod_ip(c|x_i)p(c)^{1-n}\\
\ln p(c|x)\sim \sum_i\ln p(c|x_i) + (1-n)\ln p(c)
$$

### Naive Bayes


$$
p(c|x)=\frac{p(x|c)p(c)}{p(x)}\sim p(x|c)p(c)\\
\sim \prod_ip(x_i|c)p(c) = \prod_ip(x_i,c)p(c)^{1-n}~~~~~~~~~\text{(Naive condition)}\\
\sim\prod_i\frac{N(x_i,c)}{N}p(c)^{1-n}
$$

### Semi Naive Bayes

$$
p(c|x,y)\sim p(x|c)p(c|y)\\
\sim \prod_ip(x_i|c)p(c|y) ~~~~~~~~~\text{(Semi-Naive condition)}
$$

where $p(c|y)$ will be estimated by say of neural networks.

### Hemi Naive Bayes, in more general form

When $y$ is empty, it is equiv. to the naive one.

$$
p(c|x,y_1,\cdots y_m)
\sim \prod_ip(x_i|c)\prod_ip(c|y_i)p(c)^{1-m}  ~~~~~~~~~~(Hemi-condition)\\
\sim \prod_ip(x_i|c)\prod_if_c(y_i)p(c)^{1-m}\\
\sim \prod_ip(x_i,c)\prod_if_c(y_i)p(c)^{1-m-n}
$$

## Predict

$$
\frac{p(c|x,y)}{p(c'|x,y)}= \prod_i(\frac{p(x_i|c)}{p(x_i|c')})\frac{p(c|y)}{p(c'|y)}\\
= \prod_i(\frac{p(x_i,c)}{p(x_i,c')})\frac{p(c|y)}{p(c'|y)}(\frac{p(c')}{p(c)})^n
~~~~~~~~~\text{(Semi-Naive condition)}\\
\sim \prod_i(\frac{N(x_i,c)}{N(x_i,c')})\frac{p(c|y)}{p(c'|y)}(\frac{N(c')}{N(c)})^n  ~~~~~~~~~~~~~~~~~~~~~\text{(estimate)}
$$

$$
\frac{p(c|x,y_1,\cdots, y_m)}{p(c'|x,y_1,...,y_m)}\sim ... (\frac{N(c')}{N(c)})^{n+m-1}\prod_i\frac{p(c|y_i)}{p(c'|y_i)}    ~~~~~~~~~(\text{Hemi-condition})
$$



### 0-1 cases

$$
r = \frac{p(1|x,y)}{p(0|x,y)}\sim \prod_i(\frac{N(x_i,1)}{N(x_i,0)})\frac{p(1|y)}{1-p(1|y)}(\frac{N(0)}{N(1)})^n  (Semi)\\

r \sim \prod_i(\frac{N(x_i,1)}{N(x_i,0)})\prod_i\frac{p(1|y_i)}{1-p(1|y_i)}(\frac{N(0)}{N(1)})^{n+m-1}  (Hemi)
$$

iff $r\geq 1$, $(x,y)$ is in class 1, else in class 0.



## Estimate (for continuous rv)

$p(x)\sim \frac{N(x)}{N}, N(x):$ the number of samples in a neighborhood of $x$



## Model

- Gaussian: $p(x|C)\sim N$
- Multinomial: $p(x|C)\sim M(p)=M(p_1,p_2,\cdots)$
- Bernoulli: $p(x_i|C)\sim B(p)$



### Examples

$d=\{w_k\}$

Mutinomial Model:

$(x_i=\sharp \{w\in d ,w=w_i\})\sim M(p_c)$ 

Bernoulli Model:

$x_i = w_i \in d\sim B(p_c)$ 

## Implement

```python
import numpy as np
X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 2, 2, 3, 3])
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MutlinomialNB
clf = BernoulliNB()
clf.fit(X, Y)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True) print(clf.predict(X[2:3]))
```


