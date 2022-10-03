# Zhihua Chen



# Estimating Entanglement monotones with Wootters Formula

## Introduction

### Definitions

$\rho_{AB}=\psi\otimes\psi\in H_A\otimes H_B$. (bipartite pure quantum state)
concurrence $C(\psi)=\sqrt{2(1-tr\rho_A^2)}$, where
$\rho_A=tr_B(\psi\otimes\psi)$.

#### definition

$|\psi\rangle$  is  seperable iff $C(\phi)=0$ iff $tr \rho^2_A=0$.

#### opt decomposition

$$C(\rho)=\min\{\sum_ip_iC(\psi_i):\rho=\sum_ip_i\psi_i\otimes\psi_i\},\sum_ip_i=1,p_i\geq0.$$



$\psi=\sum_{ij}\psi_{ij}\mid ij\rangle$, $L_\alpha=\mid i\rangle\langle j\mid-\mid j\rangle\langle i\mid$.

$C^2(\psi)=2(1-tr\rho_A^2)=\sum_{\alpha\beta}\langle\psi L_\alpha\otimes S_\beta\psi^*\rangle^2=\sum_{t}\langle\psi J_t\psi^*\rangle^2$. $\psi^*$ is complex conjugation.

$t=\{t_1,\cdots,t_k\}\subset\{1,\cdots,N\},u\subset\Delta^1$

$$\Delta_k(\rho,t,u)=\max\{0,\lambda_{mn}^{(1)}-\sum_{i>1}\lambda_{mn}^{(i)}\}.$$ $\lambda^{(i)}_{mn}$: sqrt of eigvals of

$$\chi=\rho(\sum_su_sJ_{t_s})\rho^*(\sum_su_s^*J_{t_s}).$$

#### observation 1

$$
C(\rho)^2\geq\frac{N}{k^2C_N^k}\sum_t\Delta_k(\rho,t,u)
$$

if $\rho$ is sep. then $\Delta_k(\rho,t,u)=0$

