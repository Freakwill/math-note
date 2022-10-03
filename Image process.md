# Poisson Editting

## Poisson Editting

$u_1, u_2$: Images, embed (part of) u1 to u2

$u_1$: source image

$u_2$: target image

### equation methods

$\Omega\subset\R^n$
$$
\min_{u_1}\|\nabla u_1-v\|\\
s.t. u_1|_{\partial\Omega}=u_2|_{\partial\Omega}
$$

where $v$: guidance vector.

$$
|N_p|u_1(p)-\sum_{q\in N_p\cap\Omega}u_1(q)=\sum_{q\in N_p\cap\Omega}u_2(q)+\sum_{q\in N_p}v_{pq}
$$

where $N_p$ is the neigherhood of $p$​.



$v_{pq}=|u2(q)-u2(p)|\lor |u1(q)-u1(p)|$​​, restricted on $\Omega$



### iterative methods

mask of gradient: m

mixed gradient: $v=m\circ\nabla u_1+(1-m)\circ\nabla u_2=m\circ\nabla (u_1-u_2)+\nabla u_2$

$Eu:=\frac{1}{2}(\|\nabla u-v\|_2^2+\|u-u_2\|_B^2)$​​ where $B$​​ is diag(multiplier)

poisson edit of $u_1, u_2$:

$$
\min Eu \sim q(B+\nabla^T\nabla, Bu_2+\nabla^Tv)= q(B-\Delta , Bu_2-w)
$$
where $w=\nabla m \cdot \nabla (u_1-u_2)+m\circ\Delta (u_1-u_2)+ \Delta u_2$

if $m$=constant, then $u_m=mu_1+(1-m)u_2,w=\Delta u_m$

 

$Eu=\frac{1}{2}(\|\nabla u-v_1\|_\Omega^2+\|u-u_2\|_B^2)$ where $B,\Omega$ is diag $B\cap \Omega\supset \partial B$ (support set)



