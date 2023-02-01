# Computing Geometry

## Bezier Curve

### Definition
Bernstein basis: $b_{i,n}=C_n^i(1-t)^{n-i}t^i$

**Definition(Bezier Curve)**: 
$$
B(\{P_i\}):=\sum_{i=0}^{n}P_i b_{i,n}
$$
where $P_i$ are control points.

Recursive definition:
$$
B(P_0):=P_0\\
B(\{P_i\}):=(1-t)B(\{P_i,i=0,\cdots,n-1\})+tP_n
$$

Operater repr:
$B(t)=(1-t +tE)^n(P_0)=(1+tT)^n(P_0)$

### Properties

1.
    $$
    B' = nB(\{P_{i+1}-P_i\})\\
    B^{(k)} = B(\{A_n^i\Delta^k(P_{i})\})
    $$
2. Graph of BC in the conv cl of CPs.
3. De Casteljau’s algorithm:
  $$
  B(t)\equiv P^n_n(t):\\
  P^r_i(t)=(1-t)P_{i-1}^{r-1}(t)+t P_i^{r-1}(t)\\
  P^0_i(t)=P_i
  $$
4. $B(\{P_k\})=B(\{P_{k}'\})$ where $P_k'=\frac{k}{n+1}P_{k-1}+(1-\frac{k}{n+1})P_k$