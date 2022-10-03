# Math of sorites

## Definition of V-spaces
$(X,\{\mathcal{V}_x,x\in X\})$, $\mathcal{V}_x$ is a non-empty family of subsets of $X$ containg $x$.

## Connection
$a,b$ are connected if
$\forall \{V_x\in\mathcal{V}_x\}\exists\{a=x_0,\cdots,x_k=b\} V_{x_{i}}\cap V_{x_{i+1}}\neq\emptyset$. ($\{a=x_0,\cdots,x_k=b\}$ witnesses the connectedness of $a,b$)

## Theorem[Dzhafarov]
Let $X$ be a V-space, $Y$ a set, $\pi:X\to Y$. If $a,b\in X$ are connected and $\pi(a)\neq\pi(b)$, then $\exists x\in X, \pi$ is not constant on any $V\in \mathcal{V}_x$.

### remark
Tolerance implies lack of Connectedness

## Definition
V-space induced by $\pi:X\to Y$ is $(X,\{\mathcal{V}_x,x\in X\})$ where $\mathcal{V}_x=\{\{y\in X|\pi(x)=\pi(y)\}=[x]_\pi\}$

## Theorem
$\pi(a)\neq \pi(b)$ then $a,b$ are not connected in $X$ deduced by $\pi$.