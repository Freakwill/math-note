# Linear Programming

## Standard form

$$
\max z=c'x\\
s.t.\begin{cases}
Ax=b\\
x\geq 0
\end{cases}
$$

### Concepts

$A=(B,Q)P, rank A= rank B​$

**Hypo.** $A$: full rank

*Base* $B$

*Base solution* $X_B=P'(B^{-1}b;0)$

*FB/BFS* $X_B\geq 0$

*OBFS* opt BFS

 

### Codes

#### optimize/Python

```python
import numpy as np
from scipy import optimize

c = np.array([2, 3, 1])
A = np.array([[1, 4, 2], [3, 2, 0]])
b = np.array([8, 6])
x1_bound = x2_bound = x3_bound =(0, None)

res = optimize.linprog(c, A_ub=-a, b_ub=-b,bounds=(x1_bound, x2_bound, x3_bound))

print(res)

# --- OUTPUT ---
#     fun: 7.0
# message: 'Optimization terminated successfully.'
#     nit: 2
#   slack: array([0., 0.])
#  status: 0
# success: True
#       x: array([0.8, 1.8, 0. ])
```

#### pulp/Python

```python
import pulp
c = [2, 3, 1]
A = [[1, 4, 2], [3, 2, 0]]
b = [8, 6]
m = pulp.LpProblem(sense=pulp.LpMinimize)
#定义三个变量
x = [pulp.LpVariable(f'x{i}', lowBound=0) for i in [1,2,3]]
#定义目标函数c[0]*x[0]+c[1]*x[0]+c[2]*x[2]
m += pulp.lpDot(c, x)

#设置约束条件
for i in range(len(a)):
    m += (pulp.lpDot(a[i], x) >= b[i])

m.solve()
print(f'优化结果：{pulp.value(m.objective)}')
print(f'参数取值：{[pulp.value(var) for var in x]}')
```



### Facts

- FS E! ==> FD: convex
- x: BFS <==> B: lin indep, (where xj>0 <==> j:B).

- LP BFS could be found in ext FD
- LP opt E! ==> x: OBFS

