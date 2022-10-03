# Projection support vector regression algorithm for data regression

X. Peng, D. Xu(2016)

## background
### regression function
$f(x)=w^Tx+b:X\subset\mathbb{R}^n\to \mathbb{R}$.
### optimization problem
$$\min \frac{1}{2}w^Tw\\
s.t. |y_i-(w^Tx_i+b)|\leq \epsilon$$

introduce slack variables

$$\min \frac{1}{2}w^Tw+\frac{C}{n}\sum_i\xi_i+\eta_i\\
s.t. -\epsilon-\eta_i\leq y_i-(w^Tx_i+b)\leq \epsilon+\xi_i,\xi_i,\eta_i\geq0$$
where $C$ is the trade-off coef.

### dual QPP:

$$\min \frac{1}{2}\sum_{ij}(\alpha_i-\alpha_i^*)(\alpha_j-\alpha_j^*)x_i^Tx_j+\epsilon \sum_i(\alpha_i+\alpha_i^*)-\sum_i y_i(\alpha_i-\alpha_i^*)\\
s.t. \sum_i(\alpha_i-\alpha_i^*)=0,0\leq\alpha_i,\alpha_i^*\leq \frac{C}{n}.$$

$w_x=\sum_{i}(\alpha_i-\alpha_i^*)x_i$.


### LS-SVR
$$\min \frac{1}{2}w^Tw+\frac{C}{2n}\sum_i\xi_i^2\\
s.t. y_i=w^Tx_i+b+\xi_i.$$

*solution.* $\sum_j(x_i^Ty_j+\frac{n}{C}\delta_{ij}\alpha_i)+b=y_i,\sum_j\alpha_j=0.$




