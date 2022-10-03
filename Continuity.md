# Thoerems

## Topology

### continuity
$f\in C(X,Y)$ iff for all $x_\lambda\to x$, there is a subnet $x_{\lambda_k}$ that $f(x_{\lambda_k})\to f(x)$. In second countable space, the net is replaced by a sequence.

*Proof.*
If $f$ is not continuous, then there is a subnet that $f(x_{\lambda_k})$ is out of a neighbourhood of $f(x)$, but $x_{\lambda_k}\to x$ hence it has a subnet $f(x_\mu)\to f(x)$, that is a contray.

Remark. In norm space, we maight take a seq. (named difference rapiddly decreasing seq.) that $\|x_{n+1}-x_n\|<2^{-n}$.

Application.


## Real Analysis
### Lemma 1
$\{f_n:C(X)\}$: bounded, $M\subset X$: countable, then $\exists \{f_k\}$ subsequence that $f_k(x)\to y, x\in M$.
### Lemma 2
$\bar{A}=X$, $\{f_k(x)\}$: Cauchy for $x\in A$, $f_k$: eq.c., then $f_k(x)$: Cauchy for all $x\in X$.

esp. $\{f_k\}$: pointedwise-Cauchy, $f_k$: eq.c., then $f_k(x)$: Cauchy for all $x\in X$.



### Thoerem

$osc_A(f):=\sup_{x,y\in A}|f(x)-f(y)|,osc_x(f):=\inf_{A\in \mathcal{T}_x}\sup_{x,y\in A}|f(x)-f(y)|.$

on Hausdorf spaces, $f$ cont. at $x$ iff $osc_x(f)=0$.