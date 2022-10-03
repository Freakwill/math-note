# Optimization

## Feasible Direction

### Linear Res.

$$
\min z=f(x)\\
s.t.\begin{cases}
Ax\geq b\\
Ex=e
\end{cases}
$$

###  Zoutendijk

#### Lemma1

$x$: FS, and $A_1x=b_1, A_2x>b_2$, then $d$ : FD of $x$ iff $A_d\geq 0, Ed=0$

where, $A=[A_1;A_2],b=[b_1;b_2]$.

LP:

$$
\min z=Df(x)^Td\\
s.t.\begin{cases}
A_1d\geq 0\\
Ed=0\\
|d|\leq1
\end{cases}
$$

#### Theorem 1

$x$:FS, and $A_1x=b_1, A_2x>b_2$, then $x$:KKT iff $0=\min LP$

 

1D:
$$
\min f(x+\lambda d)\\
0< \lambda\leq \lambda_\max\\
\lambda_\max = \begin{cases}
\infty, &d\geq 0\\
\min\{\frac{b_i}{d_i},d_i<0\},& o.w.
\end{cases}
$$


#### Algorithm

1. init. val. $X_0$

2. loop: k=0,1,...

   - $A_1x_k=b_1, A_2x_k>b_2$

   - solve LP, get $d_k$

   - if $d_k=0$, stop

   - else $\lambda =\min 1D$

     $x_{k+1}+=\lambda x_k$



### Special case

$$
\min z=f(x)\\
x\geq 0
$$

#### Theorem 1+

$x$:FS, and $x_1=0,x_2>0$, then $x$:KKT iff $0=\min LP$


$$
\min z=Df(x)^Td\\
s.t.\begin{cases}
0\leq d_1\leq 1\\
|d_2|\leq1
\end{cases}
$$



## CP

### CP, VI

$CP(F)$: $x^TF(x)=0,x,F(x)\geq 0$

$FEA(F)=\{x,F(x)\geq 0\}$



$VI(F,C): (x-x^*)^TF(x^*)\geq 0, x^*,x\in C,F:C\to \R^n$, $C$: nonempty closed covex

$VI(F):=VI(F, R^n_+)$



#### Theorem

If $C$: pointed solid closed convex cone, $F:C\to\R^n$, then

$x^*:CP(F,C,C^*)$ iff $x^*: VI(F,C)$

#### Collorary

$CP(F)=VI(F)$



#### NCP function:

$\phi(a,b)=0 \iff a=0 \lor b=0, a,b\geq 0$

CP(F): $\phi(x,F(x))=0$

Example: *Fischer-Burmeister function*

$\phi_{FB}(a,b)=\sqrt{a^2+b^2}-(a+b)$



CP equiv to

$H(x,w)=[w-F(x), w\circ x]=0, x, w\geq 0$.



#### Fixed-point iteration

x: CP(F) iff x: fp of
$$
\phi(x)=(x-\alpha F(x))_+, \alpha>0
$$


### LCP

$LCP(M,q): F(x)=Mx+q$

#### Facts

- $M$: pos then $FEA(M,q)\neq\empty$

- $M\geq0,FEA(M,q)\neq\empty$ then $S\neq\empty$ 
- $M>0$ then $S=\{x\}$ 



#### Fixed-Point Iteration

$\phi(x):=x-(Mx+q)=(1-M)x+q$ then $\phi^n(x)=(1 - M)^nx+p$.

#### Facts

$M$: sym.:

$c(\phi)\leq \|1- M\|= \max_i|1-\lambda_i|$

$c(\phi^n)\leq\max_i|1-\lambda_i|^n$.

$M\geq 0$:

$c<1$ iff $0<\lambda_i<2$



### Projection Method

in NL space, $C$: nonempty, closed, covex

$H(\alpha):=P_C(x-\alpha d),\alpha\geq0 $



#### Properties

- $y-P_C(y)^T(P_C(y)-x)\geq 0, x\in C$
- $\|P_C(z)-P_C(y)\|\leq (z-y)^T(P_C(z)-P_C(y))$