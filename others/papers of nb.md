# Notes for papers on NB

## The Optimality of NB.(2004)

### Basic

model: G: ANB(Augmented NB), Gn: NB

| model               | ANB  | NB    |
| ------------------- | ---- | ----- |
| graph               | G    | Gn    |
| classifier function | $f$  | $f_n$ |



**Definition[dep. derivative]**

dep. derivative of $x$
$$
d(x|y):=\frac{p(x|y)}{p(x)}
$$

$$
d^{c}(x|y):=\frac{p(x|y,c)}{p(x,c)}
$$

**Definition**
$$
ddr(x)=\frac{d^+(x|y)}{d^-(x|y)}
$$
**Theorem**
$$
f(x)=f_n(x)\prod_iddr(x_i)=f_n(x)D(x), x=(x_1,\cdots, x_n)
$$
**Corollary**

$f\simeq f_n\iff D(x),1\leq f(x)$ or $f(x)<1,D(x)$​



### in Gauss case

$x|c\sim N(\mu_c, \Sigma)$.

$\sum=\left(\begin{matrix}\sigma, \sigma_{12}\\\sigma_{21},\sigma\end{matrix}\right)$



**Theorem**
$$
f\simeq f_n \iff\\
1. \mu_1^+=-\mu^-_2,\mu_1^-=-\mu^+_2,\sigma_{12}+\sigma>0\\
2. \mu_1^+=\mu^-_2,\mu_1^-=\mu^+_2,\sigma_{12}-\sigma>0
$$




The less abs ratio $r=\frac{\mu_2^+-\mu^-_2}{\mu_1^+-\mu^-_1}$, the better perf. of nb.