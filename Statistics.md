# Statistics

[TOC]



## Linear Regression Analysis

### model
$$
Y=\beta_0+\sum_{i=1}^m\beta_iX_i+\epsilon,\epsilon\sim N(0,\sigma^2).​
$$

i.e. condi proba.: $Y|x_i,\beta,\sigma\sim N(\mu=...,\sigma^2)$



### samples
$Y=(Y_1,\cdots,Y_n).$ id
 assume $X$​​ is col full rank



### LSE

$$
\min_\beta \|Y-X\beta\|^2
$$

statistics:

$\hat{\beta}=X^+Y=(X^TX)^{-1}X^TY, \hat{Y}=X\hat{\beta}$.

$\hat{\sigma^2}=\frac{\|e\|^2}{n-p},e=Y-\hat{Y}$.

### statistics for LSE

#### SS table

$SST=SSE+SSR$:

$SST:=\sum_i(Y_i-\bar{Y})^2, SSE :=\sum_i(Y_i-\hat{Y}_i)^2, SSR=\sum_i(\bar{Y}-\hat{Y}_i)^2$. $MSR=\frac{SSR}{p-1},MSE=\frac{SSE}{n-p}$.

freedom: $(n-1)=(n-p)+(p-1)$.



#### test 1 $H: \beta_1=\cdots=\beta_{p-1}=0$

$F=\frac{MSR}{MSE}\sim F(p-1,n-p)$.



#### test 2 $H: \beta_k=0$

$S(\hat{\beta}):=MSE(X^TX)^{-1}$.
$\frac{\hat{\beta_k}-\beta_k}{\sqrt{S_{kk}}}\sim t(n-p)$.




#### prediction
$S(\hat{Y_0}):=MSE(1+x_0^T(X^TX)^{-1})x_0$.
$\frac{\hat{Y_0}-Y_0}{\sqrt{S}}\sim t(n-p)$.
$Y_0\sim \hat{Y_0}\pm t_{\frac{\alpha}{2}}(n-p)\sqrt{S(\hat{Y_0})}$.



### General test method

#### test $H_0$: conditions of $\beta_i$ hold

$F=\frac{SSE(R)-SSE(F)}{f_R-f_F}/\frac{SSE(F)}{f_F}\sim F(f_R-f_F, f_F)$.

$SSR(B|A)=SSE(A)-SSE(A,B)$.

### programming

[ordinary least squares](http://www.statsmodels.org/stable/regression.html)



### error analysis

MSE is est. of $\sigma^2$. $\frac{e_i}{\sqrt{MSE}}\to N(0,1),n\to\infty$.

$(q_i,e_i)$: normal prob graph.

### opt regression eq

#### Introduction
Select a subset $\{X_{i_k}\}$ from $\{X_i\}$

#### $R^2$ & Adj. $R^2$

$R^2_p=\frac{SSR_p}{SST}=1-\frac{SSE_p}{SST}, R^2_a=1-\frac{n-1}{n-p}\frac{SSE_p}{SST}$.

* $\max R_p$, add $X_i$ until $R_p$ increases insignificantly
* $\max R_a$

#### Mallows' $C_p$
$C_p;=\frac{SSE_p}{MSE(\{X_{i_k}\})}-(n-2p), EC_p\sim p$

#### Allen's $PRESS_p$

$PRESS_p=\sum_id_i^2$,
$d_i=\frac{e_i}{1-h_{ii}}$



#### Codes

```python
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])                                       
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
reg.coef_
# array([0.5, 0.5])

reg.intercept_
```

## PCA

see `ML/PCA.md`

##  CCA(Canonical corr analysis)

### Canonical  corr of pop

#### Theorem

$Cov(X)=\Sigma_{11},Cov(Y)=\Sigma_{22},Cov(X,Y)=\Sigma_{12},Cov(Y,X)=\Sigma_{21}$.

$$U_k=e_k^T\Sigma_{11}^{-\frac{1}{2}}X, V_k=f_k^T\Sigma_{22}^{-\frac{1}{2}}Y,$$
and canonical corr: $\rho_{U_k,V_k}=\rho_k$, where $\rho_k^2\in \sigma(A)$ (wrt $e_k$), and $f_k$ is eigvec of $B$,

$$A=\Sigma_{11}^{-\frac{1}{2}}\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\Sigma_{11}^{-\frac{1}{2}},B=\Sigma_{22}^{-\frac{1}{2}}\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}\Sigma_{22}^{-\frac{1}{2}}.$$

### Canonical corr (estimate)

Estimate $\hat{e},\hat{f}$ with cov of samples.

### Bartlett testing
$H_k:\rho_k=0$, $A_1\sim \chi^2((p-k+1)(q-k+1))$​

$$
W_1=\prod_{i=k}^p(1-\hat{\rho}_i^2),A_k=-(n-k-(p+q+1)/2)\ln W_1
$$



## ANOVA

### Principle

$TSS=FSS+ESS=\sum_{ij}(y_{ij}-\bar{y})^2, ESS=\sum_{ij}(y_{ij}-\bar{y}_j)^2, FSS=\sum_{j}N_j(\bar{y}_{j}-\bar{y})^2$

$H_0:\mu_j=\mu$ then $F=\frac{MSR}{MSE}\sim F(M-1,N-M)$.



## Inference

### inference quantile (sign)

* $H_0:M_p=M_0$, test: $2P(K\leq k|n',p)\leq \alpha$
* $H_0:M_p\leq M_0$, test: $P(S^-< k|n',p)\leq \alpha$
* $H_0:M_p\geq M_0$, test: $P(S^+< k|n',p)\leq \alpha$

### Cox-Staut test

$H_0$: no trend, test: $2P(K\leq k)\leq \alpha, K=\min\{S^+,S^-\}$,
$S^{+(-)}=\sharp\{i,x_i>(<)x_{i+c}\}$.



## Rank method

### Def. of Rank

The rank of $X_i$, $R_i=\sharp\{X_j \leq X_i, j=1,\cdots\}, \bar{R}_i =R_{i-1} +\frac{G_i+1}{2}$​​.

$R_i^*=\sharp\{X_j< X_i, j=1,\cdots,\lor X_j=X_i, j\leq i\}$​.

where $G_i=\sharp \{X_j=X_{(\tau_i)}\}$​.



### U stat.

$\theta - UE\to h(X_1,\cdots, X_k) - mean \to U(X_1,\cdots, X_n)$



### Wilcoxon rank test

rank of 1th group: $S_1<\cdots <S_n$, rank of 2nd: $R_1<\cdots <R_m$, $m+n=N$. $W=\sum_i S_i$ (Wilcoxon's rank sum)

$H_0: P(W=w)=\frac{\sharp\{W=w\}}{C_N^n}$

### Brown-Mood test

$H_0: med_X=med_Y$,

$A=\sharp\{X_i>M_{XY}\}\sim HG(m+n,m,t)$.

### Wilcoxon rank-sum statistic
$W_Y=\sum_iR_i, R_i=\sharp\{X_k<Y_i\}+\sharp\{Y_k<Y_i\}+1$ rank in $X_i, Y_i$.

### Mann-Whitney statistic
$W_Y=W_{XY}+\frac{n(n+1)}{2}$​;
$W_{XY}:=\sharp\{X_i<Y_j\},W_{YX}:=\sharp\{Y_j<X_i\}$​.

#### Facts
$W_{XY}+W_{YX}=mn$.



### runs test

$R$ number of runs of $X$

$H_0 $: $X_i\sim B(1,p),iid$

```R
library(lawstat)
run <- ...
runs.test(run)
```



## Max Likelihood Inf.

- Likelihood

  $l(\theta,z):=\sum_il(\theta, z_i)$

- Score function

  $l'(\theta,z)=\frac{\partial l(\theta, z)}{\partial\theta}$, $l'(\theta)=0$ (likelihood eq)

- Information matrix

  $I(\theta,z)=-H_\theta l, i(\theta)=EI(\theta,Z)$

### Fact

$z \sim p(z|\theta_0)$
$$
\hat{\theta} \to N(\theta_0, i(\theta_0)^{-1}), \hat{\theta}: m.l.e.
$$



## Dist. Est.

### empirical dist.

$F_n(t)=\frac{1}{n}\sum_i1(X_i\leq t)$

#### Theorem(Kiefer-Molfowitz)

$F_n$ max $L(x|F)$.

### KDE(kernel density est.)

$\hat{f}(t)=\frac{1}{n\lambda_n}\sum_iw(\frac{t-X_i}{\lambda_n})$

```python
gkde = scipy.stats.gaussian_kde(data) # f^ w.r.t. data
x = ...
kdepdf = gkde.evaluate(x) # f^(t)
```



## MCMC

### MH Algorithm

#### Algorithm (Metropolis-Hastings Sampling)

input: $g$

output: $\{x_t\}$ MC

1. proposal dist. $g(\cdot|x_t)$
2. $x_0\sim g(\cdot|x_t)$
3. for t=1,2,...
   - $x'\sim g(\cdot|x_t)$
   - $u \sim U(0,1)$
   - $x_{t+1}=\begin{cases}x'&u\leq A\\x_t, &o.w.\end{cases}$

where acceptance prob. $A(x',x_t)=\frac{f(x')g(x_t|x')}{f(x_t)g(x'|x_t)}$.

Special cases:

- Metropolis Sampling: $A=\frac{f(x')}{f(x_t)}$, i.e. $g(x|y)=g(y|x)$
- random walk Metropolis sampler: $g(x|y)=g(|x-y|)$
- indep MH
- componentwise MH 



*Remark.* $x_{t+1}|x',x_t\sim\begin{cases}x',& A\\x_t,&1-A\end{cases}$

*Ref.* Robert & Casella(2004)



## Normal Test

### Jarque-Bera Test

$JB:=n(\frac{S^2}{6}-\frac{(K-3)^3}{24})$

```python
statsmodels.stats.stattools.jarque_bera
```

### Lilliefors Test

```python
statsmodels.stats.diagnostic.lilliefors
```



## Corr test

Spearman rank test $H_0:X,Y$ ：uncorrelated

Spearman corr coef: $r_S=\rho(R, Q)=1-\frac{6}{n(n^2-1)}\sum_i(R_i-Q_i)^2$

Spearman T-test statistics:
$$
T = r_S\sqrt{\frac{n-2}{1-r_S^2}}\sim t(n-2)
$$

```R
cor.test(X, Y, method='spearman')  # corr test in R
# S: rank square error
# rho: spearman coef
```



```python
import rpy2.robjects as ro
ro.r['cor.test'](ro.FloatVector(y), ro.FloatVector(y1), method='spearman')  # rpy2 code
```



# Memorandum

## General

Population: $X\sim P$

samples: $X_i\sim P$ (iid)

statistics: $T(X_1,\cdots,X_n)$ function of samples

observants: $X_i=x_i$



## Statistic Model

population: $X, Y$

samples: $\{X_i\}, \{Y_i\}$ (iid from the population)

model: $Y=f(X,\theta)+\epsilon, \theta\in\R^m$, $\theta$: parameter of the model, $\epsilon$: error

condi proba: $Y|X=x, \theta, \epsilon\sim E+f(x,\theta)$

statistics(estimate of $\theta$): $\hat{\theta}(\{X_i\},\{Y_i\})$

hypothesis test is based on the distribution of $\hat{\theta}$



error: $\epsilon|x_i,y_i,\theta=Y-f(X,\theta)\sim p_i(\theta)$

MLE: $\max \prod_ip_i(\theta)\equiv \max \sum_i\ln p_i(\theta)$



## Hypothesis

$H_0: \theta\in\Theta_0$ vs $H_1: \theta\in\Theta_1$, $\Theta_0\cap\Theta_1=\emptyset$
$$
g(\theta) =\begin{cases}
P(E_1)=P(reject H_0|H_0),\theta\in\Theta_0,\\
1-P(E_2)=1-P(accept H_0|H_1),\theta\in\Theta_1
\end{cases}
$$
Principle: $P(E_1)\ll 1, P(E_1)\leq 1-P(E_2)$

1. reject domain: $T\in W$ iff reject $H_0$

2. $p$-value: $P(T=t)=p$, $p\ll 1$ iff reject $H_0$


## BIAS-VARIANCE Foundemental decomp.

### Continuous case (X is a crv)

$$
x^* := EX;\hat{x}_A:= E_X\hat{x}(X)\\
err(\hat{x}):=E_x(x- \hat{x})^2;
$$

$\hat{x}$ is any estimator of $x$ based on samples $X$,
$$
err(\hat{x},X)=err(\hat{x}(X))\\
err(\hat{x})=E_X err(\hat{x}(X))=E_X E_x(x- \hat{x}(X))^2\\
err(x^*)=E_x(x-x^*)^2=var x\\
bias(\hat{x}) := (x^*-E\hat{x}(X))^2\\
Var(\hat{x}):=E(\hat{x}(X)-\hat{x}_A)^2\\
err(\hat{x}) = bias(\hat{x}) + Var(\hat{x}) + err(x^*)
$$

for constant estimator,
$err(\hat{x}) = bias(\hat{x}) + err(x^*)$

### Discrete case (X is a drv)

$$
x^* := \argmax_x P(X=x); \hat{x}_A:=\argmax_x P_X(\hat{x}(X)=x)\\
err(\hat{x}):=P_x(x\neq \hat{x});
$$

$\hat{x}$ is any estimator of $x$ based on samples $X$,
$$
err(\hat{x},X)=P_x(x\neq \hat{x}(X))\\
err(\hat{x})=E_XP_x(x\neq \hat{x}(X))=E_xP_X(x\neq \hat{x}(X))=1-E_XP(\hat{x}(X))\\
err(x^*)=1-P(x^*)\\
bias(\hat{x}):= err(\hat{x}_A) - err(x^*)=P(x^*)-\max_x P_X(\hat{x}(X)=x)\\
Var(\hat{x}):=err(\hat{x})-err(\hat{x}_A)\\
err(\hat{x}) = bias(\hat{x}) + Var(\hat{x}) + err(x^*)
$$

for constant estimator,
$$
err(\hat{x}) = bias(\hat{x}) + err(x^*)
= (P(X=x^*)-P(X=\hat{x})) + P(X\neq x^*)
$$