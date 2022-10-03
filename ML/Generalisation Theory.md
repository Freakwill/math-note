# Generalisation Theory


## Probably Approximately Correct Learning (PAC)
Vapnik, Valiant ...: Learning theory

classifier: $h: \mathcal{X}\to \{-1,1\}$
target function: $t: \mathcal{X}\to \{-1,1\}, y=t(x)$.
predicting/generalisation error: $err(h):=P(h(x)\neq y)$​, $(x,y)\sim P$: distr. on $\mathcal{X}\times \{-1,1\}$

samples: $S=\{x_i, y_i\}\sim P, iid$,
*empirical error*: $err_S(h)=N(h(x)\neq y)/N$

$h$ is consistent with $S$ iff $err_S(h)=0$, for all $(x,y)\in S, h(x)=y$.

**PAC**: 
1. prob>= $1-\delta$ over $S$ that $\forall h\in H, err_S(h)=0$
$$
err(h)\leq \epsilon(N,H,\delta).
$$

2. prob>= $1-\delta$ over $S$ that $\forall h\in H, |\{h(x_i)\neq y_i\}|=k$
$$
err(h)\leq \epsilon(N,H,\delta).
$$

A learner consistent with $S$ (or makes $k$ errors) has a low generalisation error. (model selection problem)

sample complexity: number of samples $N(H,\epsilon,\delta)$ => $\epsilon$-level of error

statistical form: prob over $X_i\sim P ~iid$ that $X_i\in A\subset \mathcal{X}, \Rightarrow P(A^c)\leq \epsilon$ (error of A)

### finite hypo. spaces

$S$ misleads $h$​ in the $\epsilon$-level: 
$S_{h}:=\{S\in \mathcal{X}^N| err_S(h)=0\}$, where $err(h)>\epsilon$

*Fact 1*
$P(S_{h})\leq (1-\epsilon)^N\leq e^{-N\epsilon}$ where $err(h)>\epsilon$

$H$: hypo spaces (family of learners)
$S_H:=\bigcup_h S_h, S^*_H:=\{S|\exists h\in H, err_S(h)=0,err(h)>\epsilon\}\subset S_H$.

*Fact 2*
$P(S_H^*)\leq P(S_H)\leq |H|e^{-N\epsilon}$
prob at least 1-delta over $S$ that $\forall h\in H, err_S(h)=0 \Rightarrow $
$$
err(h)\leq \epsilon(N,H,\delta)=\frac{1}{N}\ln\frac{|H|}{\delta}.
$$

Hypo. test:
$H_0: err(h)>\epsilon$, $P(S_{h})$ is the p-value to reject $H_0$.

### VC dim

Let $S_{h}^{\epsilon}:=\{err_S(h)>\epsilon\}$

$P(S_{h}^{\frac{\epsilon}{2}})\geq\frac{1}{2},err(h)>\epsilon$

*Fact 3*
$P(S_H^*)\leq 2 P(\exists h\in H, S\in S_h, S'\in S_{h}^{\frac{\epsilon}{2}},err(h)>\epsilon)\leq 2P(\exists h\in H, err_S(h)=0, err_{S'}(h)>\frac{\epsilon}{2})$.

*Lemma* $P \bigcup_i A_i\times B_i \geq \min_iP B_iP\bigcup_iA_i$.

*Def(growth function)*

$B_H(N)=\max_{(x_1,\cdots,x_N)} |\{(h(x_1),\cdots, h(x_N)):h\in H\}|\leq 2^N$

$|\{(h(x_1),\cdots, h(x_N)):h\in H\}| = |H/\sim|$

$S$ is shattered iff $\{(h(x_1),\cdots, h(x_N)):h\in H\}=\{-1,1\}^N$.

$VCdim(H):=d$ iff $B_H(N)\leq (\frac{eN}{d})^d, N\geq d$

*Fact 4*
$P(S_H^*)\leq 2(\frac{2eN}{d})^d2^{-\epsilon N/2}$

*Th(Vapnik-Chervonenkis)*
$VCdim(H)=d$ => prob $\geq 1-\delta$ over $S$ that for all $h\in H, err_S(h)=0$
$err(h)\leq \epsilon(l,H,\delta)=\frac{2}{N}(d\log\frac{2eN}{d}+\log\frac{2}{\delta})$


*Th*
$VCdim(H)=d$ => for any LA exists $P$ that proba >= $\delta$ over $S$ that $h=LA(S)$, $err(h) \geq \max(\frac{d-1}{32N},\frac{1}{N}\ln\frac{1}{\delta})$.

### not full consistency
$VCdim(H)=d$ => for any $P$, prob $\geq 1-\delta$ over $S$ that for any h that $|\{h(x)\neq y\}|=k$ (makes k errors on $S$),
$$
err(h)\leq \frac{2k}{N}+\frac{4}{N}(d\ln\frac{2eN}{d}+\ln\frac{4}{\delta}),
$$
where $N>d,2/\epsilon$.
  
*structural risk minimisation*: $k\searrow,d\nearrow$

## Margin-Based Bounds
$h(x)=sign f(x)$.

$\gamma_i=y_if(x_i)$: margin of $(x_i,y_i)$ wrt $f$, denote $M_S(f)=\{\gamma_i,(x_i,y_i)\in S\}$.

margin distribution of $f$: distribution of $M_S(f)$
margin wrt $S$: $m_S(f)=\min_r\gamma_i$

### Maximal Margin bounds
*Def(cover)*
$G$ is a $\gamma$-cover of $F$ (real-valued functions) wrt $S$: for all $f\in F$ there exists $g\in G$ that $\max_i|f(x_i)-g(x_i)|<\gamma$. Let $N(F,S,\gamma)$ be the size of smallest cover.
$$
N(F,N,\gamma):=\max_S N(F,S,\gamma)
$$

*remark* $\max_i|f(x_i)-g(x_i)|$ is a semi-norm on the space of real-valued functions. It is the $\gamma$-cover under the topology.

*Th*
for any $P$, prob $\geq 1-\delta$ over $S$ and $f\in F, m_S(f)\geq \gamma$,
$$
err(f)\leq \frac{2}{N}(\ln N(F, 2N,\frac{\gamma}{2})+\ln\frac{2}{\delta}), N>2/\epsilon
$$

*Def(shattered)*
$\{x_1,\cdots,x_N\}\subset \mathcal{X}$ is $\gamma$-shattered by $F$ iff $\exists r_i\in\R,i=1,\cdots,N, \forall b:\mathcal{X}\to\{-1,1\},\exists f\in F$ that
$$
f(x_i)\begin{cases}\geq r_i+\gamma, b(x_i)=1\\
<r_i-\gamma, b(x_i)=-1.
\end{cases}
$$

fat-shattering dim(scale-sensitive VC) $fat_F(\gamma)$: the size of the largest $\gamma$-shattered subset of $\mathcal{X}$.

*Th* for any $P$, supported on $B(0,R)$, prob $\geq 1-\delta$ over $S$, $f\in L, m_S(f)\geq \gamma$,
$$
err(f)\leq \frac{2}{N}(\frac{64 R^2}{\gamma^2}\ln\frac{4N\gamma}{4R}\ln\frac{123 NR^2}{\gamma^2}+\ln\frac{4}{\delta})
$$,
where $N>2/\epsilon,64R^2/\gamma^2$.

### Soft Margin Bounds


*Ref*

N. Cristianini, J. Shawe-Taylor. An introduction to SVM and other kernel-based learning methods.