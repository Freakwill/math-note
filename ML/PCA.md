# PCA

[TOC]

## General Theory

### Notations.

cov matrix of $X:n\times p$: $\Sigma=Cov(X,X)\geq 0$. $\Gamma =e^T\Sigma e$: diag.

$\Sigma = e\Gamma e^T=\sum_i\lambda_ie_ie_i^T.$​ spectrual decomp.

$i$-PC $Y_i=Xe_i$. PCs $Y=Xe$. ($e$: eigenvectors/loadings)

$X=uDe^T, D^2=\Gamma$, SVD

reconstruction $X=Ye^T$

projection $[Y',0]e^T$

### Assumption

$X=(Y+ \mu)e^T, Y\sim N(0,\Lambda)$

### Def of PC

$$
\max Var Y_i, Y_i=Xl_i, 
\\
\|l_i\|=1, Cov(Y_i, Y_k)=0 (l_i\perp_\Sigma l_k),k<i, i=1, \cdots, p.
$$

#### Theorem

**proof.** $\lambda_{\min}\leq\Sigma \leq \lambda_{\max}$.

### Def of contribution

eigen value: $\lambda_i=Var(Y_i)=e_i^T\Sigma e_i$

contribution: $\frac{\lambda_k}{\sum_i\lambda_i},\frac{\sum_{i\leq m}\lambda_i}{\sum_i\lambda_i}$.

### decomp. with corr. *

corr matrix: $\rho=corr(X,X)\geq 0$. $\rho =e^T\Sigma e$.
contribution:

 $\frac{\lambda_i}{p},\frac{\sum_i\lambda_i}{p}$.

### PCA of sample, estimate of Y

- observe $x$
- calculate $ \hat{S}=Cov(x,x)$
- calc. $\hat{Y}=X\hat{e}, \Gamma =\hat{e}^T \hat{S} \hat{e}$.

 contribution: $\frac{\hat{\lambda}_k}{\sum_i\hat{\lambda}_i},\frac{\sum_{i\leq m}\hat{\lambda}_i}{\sum_i\hat{\lambda}_i}$.

### Codes

[statsmodels for pca](http://www.statsmodels.org/stable/generated/statsmodels.multivariate.pca.PCA.html)

[wiki for pca](https://en.wikipedia.org/wiki/Principal_component_analysis)

```python
# data: n X p - array
# not standardized, not normalized
# data: Variables in columns, observations in rows
from statsmodels.multivariate.pca import PCA

pc = PCA(data, ncomp=p, standardize=False, normalize=False) # Y=Xe
# pc.loadings: e^
# pc.factors: Y^
# pc.factors == data (demean) * pc.loadings
# pc.coeff = pc.loadings ^ -1

# principle of matrix analysis:  U S V' = D
U, s, Vh = LA.svd(np.cov(data.T), full_matrices=True)
PC = data @ Vh.H # =data @ U
# Vh.H == U == loadings
# eigenvals for lambda_i

# implement of scikit-learn
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
svd_solver='auto', tol=0.0, whiten=False)
# pca.explained_variance # eig(X^TX)/N
print(pca.explained_variance_ratio_)   # lambda_i / sum_i lambda_i
[0.9924... 0.0075...]
print(pca.singular_values_)     # sigma_i (sqrt lambda_i)
[6.30061... 0.54980...]
pca.components_ # coordinates of Xi under {Yi,i=1,2,...,p}
pca.components_.T # loadings
pca.fit_transform(X) # Yi
pca.transform(X) # Yi
pca.inverse_transform(Y) # Xi
pca.inverse_transform(pca.transform(datak)) # reconstruct/projection
```



![](/Users/william/Folders/Math Note/ML/pca.jpg)



### KL basis

$X=\sum_i\alpha_iv_i$
$$
\min E\|X-\sum_{i\leq M}\alpha_iv_i\|^2=\sum_{i> M}E|\langle X,v_i\rangle|^2=\sum_{i> M}v_i'Rv_i\\
\max\sum_{i\leq M}|\langle X,v_i\rangle|^2
$$
$R=Cov X, E|\langle X,x\rangle|^2=x'Rx$.

KL basis: solution of (2) == eig vec of $R$ (eig val in dec order) 



*remark. $X$ in random-sp. $v_i$ in lin sp.*

## ICA



## NMF

### Model

$X\sim WH$

where $X: N\times p, W:N\times r, H:r\times p, r\leq \max\{N, p\}, X,W,N$: positive

 Hypo. $x_{ij}\sim P((WH)_{ij})$, max the likelihood
$$
L(W,H)=\sum_{ij}x_{ij}\log (WH)_{ij}-(WH)_{ij}
$$

### Algorithm (Lee Seung,2001)

(Fixed-point iter.)

$\min D(X||WH)$:
$$
\begin{cases}
w_{ik} \leftarrow w_{ik}\frac{\sum_j h_{kj}x_{ij}/(WH)_{ij}}{\sum_j h_{kj}}\\
h_{kj} \leftarrow h_{kj}\frac{\sum_i w_{ik}x_{ij}/(WH)_{ij}}{\sum_i w_{ik}}
\end{cases}
$$

or $\min \|X-WH\|$:
$$
\begin{cases}
w_{ik} \leftarrow w_{ik}\frac{(VH^T)_{ik}}{(WHH^T)_{ik}}\\
h_{kj} \leftarrow h_{kj}\frac{(W^TV)_{kj}}{(W^TWH)_{kj}}
\end{cases}\\ \iff
\begin{cases}
W \leftarrow W\circ(VH^T)/(WHH^T)\\
H \leftarrow H\circ(W^TV)/(W^TWH)
\end{cases}
$$

### other

$X$: stoch. mat. then $W,H$ are assumed to be stoch. mat.

in the case, the goal is (min Cross Entropy Loss, convex opt.)
$$
\max C(W,H)=\sum_{ij}x_{ij}\log (WH)_{ij}
$$
$\chi^2=\sum_{ij}\frac{((WH)_{ij}-F_{ij})^2}{F_{ij}} \sim L$ 



#### Facts

- (W,H) is solution => (WA, BH) is solution, where $AB=1A,B$:pos
- Normalization: $(WD(W)^{-1}, SD(H)^{-1}H), S=D(W)D(H)$



#### Thoerem

NMF equiv. to pLSI, when

$X\sim p(w_i|d_j)$: stoch. mat, $W=p(w_i|z_k), H=p(z_k|d_j)$.

(diff. in algorithm!)

### Codes

```python
from sklearn.decomposition import PCA, FastICA, NMF

pca = NMF(n_components=n_components, *args, **kwargs)
pca.fit(data)
W = pca.transform(data)
H = pca.components_
```



## Frame PCA

$X=\sum_i\alpha_iv_i$, $v_i$ is a frame



$\min_{A,V} \|X-AV\|, st. A: row-sparse$