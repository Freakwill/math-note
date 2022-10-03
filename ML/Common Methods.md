[TOC]

# Incremental Learning



## Basic

### Framework

$M_i=f(D_i, M_{i-1})$

$D_i$: $i$-th data group

### Bayes' View

Bayes' Formula
$$
p(\theta |D) =\frac{p(\theta)p(D|\theta)}{p(D)}\\
=\frac{p(\theta)p(D|\theta)}{\int p(\theta)p(D|\theta)\mathrm{d}\theta}
$$
incremental form ($D\perp D'|\theta$)
$$
p(\theta |D, D') =\frac{p(D'|\theta)}{\int p(\theta|D)p(D'|\theta)\mathrm{d}\theta}p(\theta|D)
$$
where $D'$ is new data, $D$ is old data.

For supervised learing
$$
p(\theta |X, Y, X', Y') =\frac{p(Y'|\theta, X')}{...}p(\theta|X,Y)
$$
notice $(X, Y) \perp (X',Y')|\theta$



## RVM relevance vector machine

###  Basic Theory

1. Likelihood $t|x,w,\sigma^2\sim N(w\cdot\phi(x),\sigma^2)$

2. parameter-priori: $w|\alpha\sim N(0,\alpha^{-1} I)$

3. posteriori: $w|t,\alpha,\sigma^2\sim N(m,\Sigma)$ where
   $$
   m = \sigma^{-2}\Sigma \Phi^T t\\
   \Sigma = (A+\sigma^{-2}\Phi^T\Phi)^{-1}, A=diag(\alpha)
   $$

4. parameter-total: $t|\alpha,\sigma^2$
   $$
   \ln p(t|\alpha,\sigma^2)\sim -N/2\ln\sigma^2-Et-1/2\ln|\Sigma|+1/2\sum_i\ln\alpha_i
   $$

### EAP

MLE: $\max \ln p(t|\alpha, \sigma^2)$

iteration:

1. $\gamma_i=1-\alpha_i\Sigma_{ii}$
2. $\alpha_i=\gamma_i/m_i^2$
3. $\sigma^{-2}=\frac{N-\sum_i\gamma_i}{\|t-\Phi m\|^2}$



### incremental form (eq of $\alpha, \sigma^2$)

$$
\tilde{A}\sim A, \tilde{\sigma}^{2}\sim\sigma^{2}\\
\tilde{m} \sim \sigma^{-2}\tilde{\Sigma} (\Phi^T t +\Phi'^T t')=
(1+\sigma^{-2}\Sigma G')^{-1}(m + \sigma^{-2}\Sigma\Phi'^Tt')\\
\tilde{\Sigma} \sim (\Sigma^{-1}+\sigma^{-2}G')^{-1}=(1+\sigma^{-2}\Sigma G')^{-1}\Sigma\\
$$


$$
\tilde{m} \sim \sigma^{-2}\tilde{\Sigma} (\Phi^T t +\Phi'^T t')=
(\sigma^{2}+\Sigma G')^{-1}(\sigma ^2m + \Sigma\Phi'^Tt')\\
\tilde{\Sigma} \sim (\Sigma^{-1}+\sigma^{-2}G')^{-1}=(\sigma^{2}+\Sigma G')^{-1}\sigma^{2}\Sigma\\
$$


## incremental iter

$x'=\phi(x,\theta), x_0$

If $\theta$ changes, we need not restart the iteration from $x_0$

For this purpose, we should store info of the previous iter.



# Kernel Trick

*Keywords*: Riesz repr.; Bounded linear functionals; repro kernel;



### RKHS

*Def.*

$H(\Omega)$: subspace of $L^2(\Omega)$, that $\delta_x:f\mapsto f(x)$ is bounded. Let $f(x)=\langle f, K(\cdot,x)\rangle=\langle f, K_x\rangle$; repro kernel $K_x$ is **RR** of $\delta_x$, $\|K_x\|=\|\delta_x\|$. $K>0$ as function of $\Omega\times \Omega$



**Basic Fact**

$l(K_x)=v(x)$ is **RR** of $l$, $\langle v, v'\rangle=l'_yl_x(K(x,y))$

RK: $\langle K(\cdot,x'), K(\cdot,x)\rangle=K(x,x')$: $\Omega\times\Omega\to \C$



*Def*

$K$: kernel on $\Omega$ iff $K(x,y)=\langle\Phi(x), \Phi(y)\rangle$, where feature map $\Phi:\Omega\to\mathcal{H}$



**Thm(existance of RKHS)**: $K>0$ <=> E! RKHS $H$ with RK $K$, i.e. cl-span of $K(\cdot, x)$, where $f=\sum_i\alpha_iK_{x_i},g=\sum_i\beta_iK_{x_i}\in H$,
$$
\langle f, g\rangle = \sum_{ij}\alpha_i\beta_j K(x_i,x_j).
$$
$K\geq 0$ <=> $K$: kernel; $\mathcal{K}\phi_i(x):=\int\phi_i(x')K(x',x)=\lambda_i\phi_i(x)$. Feature map $\Phi(x):=\{\sqrt{\lambda_i}\phi_i(x)\}$ (Mercer features)

$H_k:= \overline{\mathrm{span}}\{ \phi_i(x)\}$, with ob $\{\sqrt{\lambda_i}\phi_i\}$ => $K$: RK of $H_k$



*Example 1*

RKHS of wavelet (sub sp. of $L^2(G)$): $F(a)=\langle f,\pi(a)\psi\rangle$, $K(a,b)\sim \langle \pi(b)\psi,\pi(a)\psi\rangle$ where $\Phi(a)=\pi(a)\phi$

RKHS of dual wavelet: $F(a)=\langle f,\pi(a)\psi\rangle$, $K(a,b)\sim\langle \pi(b)\tilde\psi,\pi(a)\psi\rangle$



*Example 2*

char set: $\Sigma$

string : $\Sigma^*$

$\Phi(s)_u=\sum_{s(i)=u}\lambda^{l(i)}, u\in \Sigma^n$ where $i$ is the indexes, $l$ is the length of their range

 

**Lemma**

In $\R^n$, matrix $G\geq (>)0$ iff $\exists \{x_i\in H\}, G=\{\langle x_i, x_j\rangle\}$ ($\{x_i\}$ lin. indep.)

$K\in C_C(X\times X)$ => $K\geq 0$ iff for all $\{x_i\in X\}$, $\{K(x_i,x_j)\}\geq 0$. => $K(x,y)=\langle\Phi(x),\Phi(y)\rangle$, $\Phi:X\to\mathcal{H}$(feature sp.) $\Phi(x):=\{\sqrt{\lambda_i}\phi_i(x)\}$ (Mercer features)



#### Duality with GP

$K(x,x')=E(u(x)u(x')), u\in GP$: cl-span of $u(x)$

$u(x_i)\mapsto K_{x_i}$: iso $GP\to H_k$

