# ML

## DR

### NMF

#### Applications

1. text data clustering
2. graph data representation
3. face recognization
4. blind signal
5. DNA analysis
6. light spectrum

#### Categories

1. 加速算法：投影梯度法、牛顿法
2. 约束NMF: 图正则NMF、正交约束NMF、半监督NMF、robust
3. 结构NMF：加权NMF、非负矩阵三分解
4. 推广NMF：非负张量分解、半非负矩阵分解



### subspace sep

1. 代数方法
2. 迭代方法
3. 统计方法
4. 谱聚类方法



### sparse rep.

$$
\min \|Z\|_1, \lambda\|E\|_{2,1}\\
st. X=XZ, diag Z=0
$$

### RPCA

$$
\min \|Z\|_* + \lambda \|E\|_1\\
st. X= Z+ E
$$

#### low-rank rep. (LRR)

$$
\min \|Z\|_* + \lambda \|E\|_{2,1}\\
st. X= XZ+ E
$$

#### Matrix Completion(MC)

$$
\min \|X\|_*\\
st. P_\Omega X=P_\Omega Z
$$

- 基于半正定规划，CVX

- 基于软阈值算子，SVT、FPCA、ALM、APG

- 流形优化，OptSpace、SET

  

### NL DR

- ISOMAP
- LLE
- LE map
- LTA
- SC
- KPCA,KLDA



### SSL

#### SSC

- 约束，ML、CL
- 距离, Xing
- 结合约束与距离，Bilenko



### NPKL

- MKL
- NPKL: OSK, NPK,LRK, TSK




## Spectral Clustering

### AP

Frey, Dueck, 2007
$$
\arg\max\sum_i s(x_i,z(x_i))
$$
$z:X\to X$代表元映射

*算法*：

竞争规则：

$r_{ik}:=s_{ik}-\max_{j\neq k}\{a_{ij}+s_{ij}\}$, 代表元$k$对$i$的责任度

迭代规则：

$a_{ik}:=\min\{0,r_{kk}+\sum_{j\neq i,k}\max\{a_{ij}+s_{ij}\}\},a_{kk}=\sum_{j\neq k}\max\{0,r_{jk}\}$, $i$选$k$的可信度

选择：

$k^*_i=\arg\max a_{ik}+r_{ik}$



Preferences $P$:

$r_{kk}=P_k-\max_{j\neq k}\{a_{kj}+s_{kj}\}$



### Nystrom Method



### SOM

Algo.

1. 归一化：样本、权重归一化

2. 对每个样本

   - 计算$\langle x, w_j\rangle$, 得到获胜神经元$j^*$

   - 调整$w_{ij}=w_{ij}+\eta(x_i-w_{ij}), j \in N_{j^*}$.

