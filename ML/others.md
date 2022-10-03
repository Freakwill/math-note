# other topics of ML

### Levenshtein distance.

#### Definition

$L(a,b)=\min |p|,p:a\to b$ seq. of remove/insert/replace

#### Recursive

$$
L_{a,b}(i,j)=\begin{cases}
0, i=j\\
i, j=0\\
j, i=0\\
\min\begin{cases}
L(i-1,j)+1,& remove a_i\\
L(i,j-1)+1, & insert b_j\\
L(i-1,j-1)+1(a_i\neq b_j),& replace a_i/b_j
\end{cases}
\end{cases}
$$

$$
L(a,b)=\begin{cases}
0, a=b\\
|a|, b=''\\
|b|, a=''\\
\min\begin{cases}
L(a',b)+1,& remove a_m\\
L(a,b')+1, & insert b_n\\
L(a',b')+1(a_m\neq b_n),& replace a_m/b_n
\end{cases}
\end{cases}
$$

#### Properties

$L$: distance on strings

$L(cad,cbd)=L(a,b)$

$L(a,b) \leq L(a,bc), |a|\leq|b|$

$L(a,b)\geq ||b|-|c||$



### C-m-kNN

$C=\{C_1,\cdots,C_m\}$
$$
G(C):=\sum_j\|y_j-h_C(x_j)\|+\lambda\sum_j\rho(x_j,\mu_C(x_j)),\\
h_C(x)=\frac{\sum_{i\in N_C(x),k}y_i}{k},\mu_C(x_j)=\frac{\sum_{i\in N_C(x)}x_i}{|N_c(x)|}
$$
$N_C(x) =\arg\min_{C_k} \{x-\mu_k\}, \mu_k$: center of $C_k$



$\nu_C(x)=\frac{\sum_{i\in N_C(x)}y_i}{|N_c(x)|}$ 
$$
F(C):=\sum_j\|y_j-\nu_C(x_j)\|+\lambda\sum_j\|x_j-\mu_C(x_j)\|
$$



## LL

### Definition

input space transform $g:I\to I'$

Thrun(1996)
$$
E=\sum_{D_i}E_i, E_i(g)=\sum_{<x, y=1> \in D_i}(\sum_{<x', y'=1> \in D_i}\|g(x)-g(x')\|-\sum_{<x', y'=0> \in D_i}\|g(x)-g(x')\|)\\
=:V_1(g)-V_{0,1}(g)
$$
$g$ = argmin $E$



Thrun sample distance: $d(x,x')=\delta(y,y')$, where $(x,y),(x',y')\in D_i$.

*Remark* $V_1(g)=\sum_{<x, y=1> \in D_i}\sum_{<x', y'=1> \in D_i}\|g(x)-g(x')\|$ so on.

