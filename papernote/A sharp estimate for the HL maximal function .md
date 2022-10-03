# A sharp estimate for the HL maximal function 
L. Grafakos (1996)


### Definition
$Mf(x):=\sup_{\delta>0}\frac{1}{2\delta}\int_{x-\delta}^{x+\delta}|f(t)|dt$.

Grafakos functions ($P$):
$f\in S(R\setminus\{0\}), f(x)=\frac{1}{|x|^{2p}}, |x|\ll1$, $f$ is convex except at 0.

mean value function family:
$$\xi_x(t)=\begin{cases}\frac{1}{2t}\int_{x-t}^{x+t}f(u)du,& t>0,\\
f(x), & t=0.
\end{cases}\in S(R\setminus\{|x|\})\cap C$$

### Fact
$\xi_x'(t)=\frac{(f(x+t)-f(x-t))/2-\xi_x(t)}{t}$
$\xi_x'(t)\geq 0, t\in(0,|x|], >0, t\to |x|$. Thus $\xi_x\nearrow$ in neighborhood of $(0,|x|]$.

$\delta(x):=\max B_x, B_x=\arg\max \xi_x$. Grafakos mapping

$\delta(x)\in S(R\setminus\{0\})\cap C$ [IFT]


### Lemma 1. 
For $x\neq0$, $Mf(x)=\frac{f(x+\delta(x))+f(x-\delta(x))}{2}$, $Mf'(x)=\frac{f(x+\delta(x))-f(x-\delta(x))}{2\delta(x)}$.

$l:$ tangent line to $G(Mf)$ at $x$, $l$ intersects with $G(f)$ at $x+\delta(a), x-\delta(a)$.

### Lemma 2.
If $x>0$, then $\delta'(x)>1$, if $x<0$, then $\delta'(x)<-1$. $Mf\in P$.


