# 主成分分析PCA

[TOC]

**关键字** PCA SVD Karhunen-Loeve-基

## 一般原理

### 记号

 $X$ ($p$维随机变量) 协方差矩阵: $\Sigma=Cov(X,X)\geq 0$. 

$\Gamma =e^T\Sigma e$ 正定矩阵对角化（特征值降序排列）

$\Sigma = e\Gamma e^T=\sum_i\lambda_ie_ie_i^T.$ 谱分解形式



（wlog, 设$X$ 0 均值。）

$X=uDe^T, D^2=\Gamma$, SVD

主成分 $i$-PC: $Y_i=Xe_i$. PCs: $Y=Xe$. ($e$: 特征向量/负载)

分解$Y=Xe $

重构 $X=Ye^T$

滤波-重构 $X'=[Y',0]e^T$



### 统计学解释

#### 假设

随机向量$X$服从混合正态分布, 即$\mu+Ye^T, Y_i\sim N(0,\lambda_i)$($Y_i$相互独立)。

| $X$显变量 | $Y$隐变量（相互独立） |
| --------- | :-------------------- |
| 词语      | 主题                  |
| 面部轮廓  | 特征脸                |
| 行为特点  | 人格特质              |

*注*$Y$的维数不应超过$X$, 但应尽量包含$X$的信息:scorpion:



#### PC 定义

$$
\max_l Var Y_1, Y_1=Xl_1,
\\
s.t. \|l_1\|=1
\\
\max_l Var Y_i, Y_i=Xl_i,
\\
s.t. \|l_i\|=1, Cov(Y_i, Y_k)=0 (l_i\perp_\Sigma l_k),k<i, 1<i<p.
$$

**定理** (1) 的解是$\Gamma =e^T\Sigma e$ (正定矩阵对角化)中的$e$，第$i$列为$l_i$，称为主成分，和$Y_i$相对应。

**证明.** (略)

*注*  PCA对Gaussian随机变量最合适，因为Gaussian分布的熵完全由方差确定，方差越大熵越大，包含的信息就越多。:paw_prints:

### 线性代数解释

向量组$X=\{X_1,\cdots, X_p\}$, 用正交向量组$Y=\{Y_1,\cdots, Y_q\},q\leq p$表示

$X=YA$, $X^TX=Y^TY$且误差$\|X-YA\|_2$最小。

### KL 展开

$X$: 随机向量，$\phi_i$: 标准正交基 (p 维线性空间)

正交分解$X= \sum_i\alpha_i\phi_i$, $\alpha_i=\langle X,\phi_i\rangle$是随机变量

#### Karhunen-Loeve-基

**定义（KL-基）**

$\phi_i$:KL-基 := $\arg\min_{\phi_i} E\|X- \sum_i\alpha_{i<m}\phi_i\|_2, m\leq p$ 



**定理** $\phi_i$是KL-基 iff $\Gamma =\phi^T\Sigma \phi$ (正定矩阵对角化) 

$X\sim\sum_i\alpha_i\phi_i$

:electric_plug: *联系*

$X=Ye^T=\sum_i\alpha_i\phi_i$: $Y_i$是隐变量，也是系数；$e$是KL基（特征向量）



### 概率PCA

$$
x = Wh +b + z, h\sim N(0, 1), z\sim N(0,\sigma^2)
$$

显变量：$x$

隐变量：$h$

参数：$b, W,\sigma^2$

### 相关概念

#### 百分比定义（contribution）

特征值:

$\lambda_i=Var(Y_i)=e_i^T\Sigma e_i$

贡献百分比，累计贡献百分比：

$\frac{\lambda_k}{\sum_i\lambda_i},\frac{\sum_{i\leq m}\lambda_i}{\sum_i\lambda_i}$.

百分比：衡量主成分回复原始信息的贡献

#### 相关系数矩阵（对$X$标准化）

相关系数矩阵: $\rho=corr(X,X)\geq 0$. $\rho =e^T\Sigma e$.
百分比: $\frac{\lambda_i}{p},\frac{\sum_i\lambda_i}{p}$.

*注* 特征值和百分比会受标准化影响，标准化常用于分量不同量纲情形

### 样本PCA,  Y估计

#### PCA算法

- 观察值$x\sim X$

- 计算样本协方差$\hat{S}=Cov(x,x)$

- 计算$\hat{e}, \Gamma =\hat{e}^T \hat{S} \hat{e}$，作为$e$的估计，变换得到$\hat{Y}=x\hat{e}$，以及

  百分比估计 $\frac{\hat{\lambda}_k}{\sum_i\hat{\lambda}_i},\frac{\sum_{i\leq m}\hat{\lambda}_i}{\sum_i\hat{\lambda}_i}$.
  
  

## Python 实现

[scikit-learn实现](https://scikit-learn.org/stable/modules/decomposition.html#pca)

```python
# implement with scikit-learn
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)

PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
svd_solver='auto', tol=0.0, whiten=False)

print(pca.explained_variance_ratio_)   # lambda_i / sum_i lambda_i
[0.9924... 0.0075...]
print(pca.singular_values_)     # sigma_i (sqrt lambda_i)
[6.30061... 0.54980...]

pca.components_  # the pcs (sorted by lambda_i)

C=pca.transform(A) # components
pca.inverse_transform(pca.transform(A)) # reconstruct
```



[statsmodels 实现](http://www.statsmodels.org/stable/generated/statsmodels.multivariate.pca.PCA.html)

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

```

numpy 实现

```python
# principle of matrix analysis:  U S V' = D
U, s, Vh = LA.svd(np.cov(data.T), full_matrices=True) # U == V
PC = data @ Vh.T
# V == loadings
# eigenvals for lambda_i
```



## PCA 变种

### IPCA

用于大数据的PCA

`ipca = IncrementalPCA(n_components=n_components, batch_size=10)`

### KPCA

核技巧

`kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)`

### SPCA

提取稀疏成分

`transformer = SparsePCA(n_components=5, normalize_components=True, random_state=0)`

## 后记

PCA的关键是估计协方差矩阵$\hat{S}$，其余是 Algebraic.

1. 样本协方差估计
2. 利用$X$参数分布，给出更好的无偏估计



## 应用

### 信息压缩

$A=CV^T\sim C_1V_1^T$

1. 保存 $C_1$, 压缩$A$为$V_1^T$
2. 表示 $\xi=C_1^+x$，
3. 计算$\xi$与$V_1^T(j)$的距离，进行归类.

### 例子

特征脸、文本主题（LSA）

#### 个人作品

[图像pca](https://gitee.com/williamzjc/image-pca)

## 机器学习类比

|        | 无监督学习-隐变量模型 | 监督学习-无隐变量 |
| ------ | --------------------- | ----------------- |
| 连续值 | 降维，如PCA           | 回归              |
| 离散值 | 聚类                  | 分类              |



## 文献

[PCA wiki](https://en.wikipedia.org/wiki/Principal_component_analysis)