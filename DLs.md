# DLs

## ALC

1991, Schmidt-Schauss,Smolka

### Definition

#### Grammar

$N_C,N_R$:不相交原子概念集、原子关系集

1. $A\in N_C$是ALC概念
2. $R\in N_R$, $C,D$是概念，则$\lnot C, C\sqcup D, C\sqcap D,\exists R.C,\forall R.C$是概念
3. $\top, \bot$是概念

#### Semantics

Tarski式语义

*Depth*
$$
\begin{cases}
d(\lnot C)= d(C)\\
d(C\sqcup D), d(C\sqcap D)= \max\{d(C),d(D)\}+1\\
d(QR.C)=d(C)+1
\end{cases}
$$
$d(x:C)=d(C),d(C\sqsubset D)=\max\{d(C),d(D)\}$.

$d(A)=1, A\in N_C,d(R)=1, R\in N_R$.

$d(\{A_i\})=\max d(A_i)$

### 知识库

#### General TBox

$I\vDash C\doteq D \iff C^I=D^I$

#### Acyclic TBox

$A ref B$ iff $A\doteq C, A\sqsubset C$, $C$包含$B$

1. 不存在$A ref B$
2. 原子概念定义/特化在左端只出现一次

#### ABox

$a\in O_I$个体集

$a:C,(a,b):R$



### 推理



#### 空TBox 推理

*Tableau Algo*

利用一致性不变规则实现完备化


tab(A):

0. init: $A_0=\{x_0:C_0\}, S_0={A_0}$

1. $\sqcap$-rule:
condition: $x: D \sqcap E$ in A, $x:D$ or $x:E$ not in A
op: $A'=A \cup \{x:D,x:E\}$

2. $\sqcup$-rule:
cond.: $x: D\sqcup E$ in A, $x:D,E$ not in $A$
op: $A'=A \cup \{x:D\}, A''=A\cup\{x:E\}$

3. $\forall$-rule:
  cond.: $x: \forall R.D,  y (x,y):R$ in $A$ and $y:D$ not in $A$
  op: $A'=A\cup\{y:D\}$

4. $\exists$-rule:
  cond.: $x: \exists  R.D$ in A, no y $(x,y):R$ and $y:D$ in $A$
  op: $A'=A\cup\{z:D, (x,z):R\}$, $z$ is new
  
5. o.w.

  op: close

$S_i\to S_{i+1} = S_i + T_i, |T_i|<|S_i|, d(T_i)<d(S_i)$



*Remark.* label $\mathcal{L}(x)=\{C, x:C\}$

##### Fact

Rule1-4, 不改变一致性




$A$: 完备 iff Algo. 条件都不满足

$A$: 封闭，如果包含冲突$\{x:D,x:\lnot D\}$, 否则开放

$S:$ 开放，若$\exists A\in S$开放

##### Thm

Algo. 可终止($S_n$完备), 不改变一致性



4' Trace $\exists$-rule:

cond.: $x: \exists R.D$ in A, no y that $(x,y):R'$ in A
op: $A'=A\cup \{z:D, (x,z):R\}$, $z$ is new

*Trace-Tableau Algo*:

(Tableau Algo1-3)* -> 4' -> ...

