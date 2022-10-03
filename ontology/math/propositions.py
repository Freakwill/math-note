#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from base import *


# analysis
with math.Relm('Analysis'):
    kroneckerLemma = math.Lemma(title='Kronecker Lemma', contributor='Kronecker') \
    @ '''
    xi in R, bi in R+ -> infty. Sn=sum_n xi, rn = sum_n bixi.
    If sn->s < infty, then rn/bn -> 0.
    ''' \
    >> 'Abel sum theorem'

    diniTheorem = math.Theorem(title='Dini Theorem', contributor='Dini') \
    @ """
    If f_n(x): C(X) ->0 mon., X: compact, then f_n => 0.
    """ \
    >> """1.define E_{n,e}={f_n(x)<e} (Dini Covering)
    2. f_n(x_n) = sup f_n(x) -> a, x_n->x: Dini seq.
    """ \
    ^ 'f_n(x)>=0: upper-semi cont.'
    diniTheorem.wiki = "Dini's theorem"

## inequalities
with math.Relm('Inequalities'):
    friedrichsInequality = math.Formula(title='Friedrichs Inequality', contributor='Friedrichs') \
    @ """
    int_Omega |u|^2  <= C(Omega) int_Omega |Nabla u|^2, u in C1(cl Omega) and u|(bd Omega)=0
    """ ^ "in 1D, u(x) == int[a, x] u'"

    dipe = math.Formula(title='IPE of derivatives') \
    @ """
    int_Omega1 |Nabla u|^2 <= C(Omega, Omega1) int_Omega u^2 + int_Omega f^2, D2u=f in L2, u in C2 & L2(Omega), cl(Omega1) in Omega
    """ \
    >> "insert a bump function phi|Omega1=1"
    youngInequality = math.Formula(title='Young Inequality', contributor='Young') \
    @ """
    f:[0,oo] strictly inc, => ab<=int_0^af(x)+int_0^bg(x), a,b>=0, g = inverse of f;

    forall a, b>=0, ab <= a^p/p + b^q/q where 1/p+1/q=1, p,q>0
    = holds when a^p=b^q

    *remark* forall a, b>=0, a^pb^q <= pa + qb where p+q=1, p,q>0
    = holds when a=b
    """

    chernoffInequality = math.Formula(title='Chernoff Inequality', contributor='Chernoff') \
    @ """
    Xi ~ B(pi); X = mean_i Xi, m = mean_i p_i, i=1...n
    => P(X>=(1+d)m) <= exp(-mnd^2/(2+d)); P(X<=(1-d)m) <= exp(-mnd^2/2)

    more general, Xi in [a,b]
        => P(X>=(1+d)m) <= exp(-2nm^2d^2/(b-a)^2); P(X<=(1-d)m) <= exp(-2nm^2d^2/(b-a)^2)
        or P(X>=m + e) <= exp(-2ne^2/(b-a)^2); P(X<=m - e) <= exp(-2ne^2/(b-a)^2)
        or P(|X-m|>=e) <= 2exp(-2ne^2/(b-a)^2)
    """ \
    >> "Bernstein's idea. apply Markov ineq. to e^tX, for any t>0." \
    ^ """the bounds are called Chernoff bounds
    in the relm of statistical learning.
    x ~ D, S iid~ D, prob over S <= exp(-2ne^2)
    P(l(x))>=PS(l(x))+e where l(x):{0,1}.
    It holds for PS(l(x))>=P(l(x))+e

    l in L, |L|<oo
    prob over S <= 1-d, d=|L|exp(-2ne^2)
    P(l(x))<=PS(l(x))+e where l(x):{0,1}, for l in L.
    """ \
    ^ "See Hoeffding's Inequality"

    hoeffdingTheorem = math.Formula(title="Hoeffding's Inquality", contributor="Hoeffding") \
    @ """ai<=Xi<=bi; Sn = sum_i Xi; Xi: indep rv
    => P(Sn-ESn>=t) <= exp(-2t^2/sum(bi-ai)^2)
       P(|Sn-ESn|>=t) <= 2exp(-2t^2/sum(bi-ai)^2)
    """ \
    >> "Hoeffding's Lemma" \
    ^ "W. Hoeffding. Probability inequalities for sums of bounded random variables, 1963."

# real analysis
with math.Relm('Real Analysis'):
    fatuLemma = math.Lemma(title='Fatu Lemma', contributor='Fatu', applied_in=['limit of integral',]) \
    @ 'int liminf_n f_n <= liminf_n int f_n'

    lebesgueDominatedConvergenceTheorem = math.Theorem(title='lebesgue Dominated Convergence Theorem', contributor='Lebesgue', applied_in=['limit of integral',])
    fatuLemma.Prove = [lebesgueDominatedConvergenceTheorem]

    mooreAronszajnTheorem = math.Theorem(title='Moore-Aronszajn Theorem', applied_in=['kernel trick']) \
    @ 'K:X * X->R is a reproducing kernel on H (unique) iff K > 0 is sym and pd' \
    >> """
    Let Kx=K(x,.), H0 = span{Kx}: Hilbert space under <sum_i aiKxi, sum_i biKyi>=sum_{ij}aibjK(xi,xj), esp. <Kx, Ky>=K(x,y)
    H := cl H0.
    """
    mercerTheorem = math.Theorem(title='Mercer Theorem', applied_in=['kernel trick']) \
    @ 'K>=0 (cont.) sym. iff K(x,y)=sum_i lambda_iei(x)ei(y), ei: (cont.)'

# harmonic analysis
with math.Relm('Harmonic Analysis'):
    vitaliCoveringLemma = math.Lemma(title='Vitali Covering Lemma', contributor='Vitali', applied_in='Hardy-Littlewood Inequality') \
    @ "B1, ..., Bn: balls, E! disjoint sub-collection B1',...Bm' that Union_iBi <= Union_j(3Bj')"
    vitaliCoveringLemma.remark = "B1,..., countable balls (max radius<infty), E! such with scale s>3;"

    besicovichCoveringLemma = math.Lemma(title='Besicovich Covering Lemma', contributor='Besicovich')
    vitaliCoveringLemma.alsosee = besicovichCoveringLemma

    hardyLittlewoodTheorem = math.Theorem(title='Hardy-Littlewood maximal Theorem') \
    @ "M: Weakly bounded"

    haarTheorem = math.Theorem('Haar Theorem', contributor='Haar') \
    @ "in LCG X, E! mu: Haar measure"
    haarTheorem.remark = "Haar only proved compact case."

with math.Relm('Functional Analysis'):
    banachExtensionTheorem = math.Theorem(title='Banach Extension Theorem', contributor='Banach') \
    @ '''X: R-lin.sp, p: X-semi-norm, X_0 < X, f0: L(X0), f0(x)<= p(x), then
    i) f(x) <= p(x), x: X; (extension of f0)
    ii) f(x) = f0(x), x: X0.
    If X: C-lin.sp, |f0|<=p   ==> |f| <= p.
    ''' \
    >> 'f ext f0 is a po; require Zorn Lemma'

    hahnBanachExtensionTheorem = math.Theorem(title='Hahn-Banach Extension Theorem', contributor='Banach') \
    @ '''X_0 < X: LNS, f0: L(X0), then exists f:L(X)
    i) f(x) == f0(x), x: X0; (extension of f0)
    ii) ||f|| = ||f0||.
    '''

    banachExtensionTheorem.Prove = [hahnBanachExtensionTheorem]

    banachSteinhausConvergenceTheorem = math.Theorem(title='Banach-Steinhaus Convergence Theorem') \
    @ '''X: LNS, Y: Banach, M: X complete, Tn, T: L(X,Y)
    i) {Tn} bounded
    ii) Tnx -> Tx, x in M (s)
    => Tn -> T (norm)
    ''' \
    ^ """X: LNS, M: X complete, fn, f: X*
    i) {fn} bounded
    ii) fnx -> fx, x in M (s)
    => fn -> f (norm)
    """

    fredholmSurjectTheorem = math.Theorem(title='Fredholm Surject Theorem') \
    @ '''In Hilbert space,
    TT*=T*T, exists m>0, all x ||Tx||>m ||x||, then T: H<->H, 0 in rho(T)
    '''

    stoneWeierstrass = math.Theorem(title="Stone Weierstrass Theorem") \
    @ """F is a subalgebra (/sublattice) of C(X) (real valued),
    contains 1, separates points in X (/has interpolation property),
    then F is dense in C(X), where X: Compact Hausdorff.
    """ \
    >> """fact: h:X -> R, s, t: X, exists f:F, f(s)=h(s), f(t)=h(t). f: S-W helper
   for each s,
   U_t := {r:X, ft(r)>h(r)-epsilon}: open cover of X (determined by s, h)
   gs = sup(ft1 ... ftn): F; for t: X, gs(t)>h(t)-epsilon, gs(s)=h(s);

   V_s := {r:X, gs(r)<h(r)+epsilon}: open cover of X (determined by s)
   g = inf(gs1 ... gsn)

   idea: find finite f that f(r)<h(r)+epsilon
    """ \
    ^ "F is a sub linear space, contains 1, separates points, then has interpolation property." \
    ^ "Application: {f(tx)} comp if f(tx) strictly monotomic for one t, f(0)>0, f(tx)f(sx) in span{f(tx)}"

    laxMilgramTheorem = math.Theorem(title='Lax-Milgram Theorem') \
    @ """
    U, V: Hilbert sp.
    Bounded bilinear functional b: U X V -> C, coercive: b(u,u) ~ |u|^2
    => f: V, E!1 uf, b(uf, v)= <f, v> for all v |uf| ~ |f|
    """ \
    >> """
    Note: bounded bilinear(b(u,u) <~ |u|^2) => bounded op b(u,v)=<Au, v>

    Babuška-Lax-Milgram
    U, V: Hilbert sp.
    Bounded bilinear functional b: U X V -> C, weakly coercive:
    1. sup_{|v|=1} b(u,v)>c|u|, c>0
    2. sup_{|u|=1} b(u,v)>0
    => f: V*, E!1 uf, b(uf, v)= f(v) for all v, |uf|<1/c|f|
    """


# stachastic process
with math.Relm('Stachastic Process'):
    '''Notations:
    I[0,t]f: Ito integral of f on [0,t]
    '''

    itoIsometry = math.Proposition(title='Ito Isometry')
    itoIsometry.content = '''
    Ito integral: isometric
    '''

    itoIsometry.proof = 'approximation argument'

    martingale = math.Definition(title='Martingale') \
    @ '''Mt: Martingale(cont.) / Ft iff Mt: Stochastic Process(cont.) Mt: L1, E(Ms|Ft)=Mt, s>t
    '''
    martingale.remark = 'A project sequence: Mt < Ms, t<s'

    doobMartingaleInequality = math.Theorem(title='Doob Martingale Inequality', contributor='Doob', applied_in='Ito integral')
    doobMartingaleInequality.content = '''If Mt: Martingale(continuous), then forall p>=1, T>=0, lambda>0
    P(sup t |Mt|>=lambda) <= 1/lambda^p E|Mt|^p
    '''
    doobMartingaleInequality.remark = ''' f: L^2/Ft, exists cont. Martingale/Ft Jt = I[0,t] f, a.s,
    '''

    itoRepresentationTheorem = math.Theorem(title='Ito Representation Theorem')
    itoRepresentationTheorem.content = ''' F in L^2(T), exists f: L^2, F=EF+I[0,T] f
    '''

    martingaleRepresentationTheorem = math.Theorem(title='Martingale Representation Theorem')
    martingaleRepresentationTheorem.content = ''' Mt: Martingale, unique f: L^2, F=EMt+I[0,t] f
    '''

with math.Relm('Differentable Manifolds'):
    dpsiRepresentationTheorem = math.Proposition(title='Representation of dpsi') \
    @ '''psi: M-> N, m|->psi(m), dpsi: Mm->Npsi(m),
    mat(dpsi)=Jpsi, under nat. basis..
    
    esp. psi: M-> R, crd(dpsi*)=Dpsi, under dual basis.
    '''


with math.Relm('Probability Theory'):
    suffstatTheorem = math.Theorem(title='Hoeffding inequality', contributor='Hoeffding') \
    @ '''Z1 ... Zn: indep rv in [a,b], M = mean of Zi
    P(M-EM >= t)<= exp {-2nt^2/(b-a)^2}; P(M-EM <= -t)<= exp {-2nt^2/(b-a)^2}
    ''' \
    >> "Use Hoeffding’s lemma: E(exp (Z-EZ))<= exp {(b-a)^2/8}, where Z in [a, b]"


with math.Relm('Statistics'):
    suffstatTheorem = math.Theorem(title='Sufficient Statistic Representation', contributor='Fisher–Neyman') \
    @ ''' T(X) est. theta, X ~ p_theta,
    T(X): suff iff p_theta(x) = g_theta(T(x))h(x), g_theta: T-measurable, h: pdf.
    '''


with math.Relm('Ergodic thorems'):
    poincareRecurrenceTheorem = math.Theorem(title='Poincare Recurrence Theorem', contributor='Poincare') \
    @ r"""T: X->X, X: Probability Space, mu_T=mu
    a.e. x:A, exists {n_i} T^{n_i}x : A, mu A>0.
    """ \
    >> 'Construct a complemental set'

    birkhoffKhinchinTheorem = math.Theorem('birkhoffKhinchinTheorem', title='Birkhoff Khinchin Theorem', contributor='Birkhoff-Khinchin') \
    @ r"""as in poincareRecurrenceTheorem,
    a.e. x, exists f'(x)=lim_n 1/n sum_k(fT^kx):L1, where f:L1, and int f=int f'.
    """ \
    >> 'Construct a T-algebra'

    vanderWaerdenTheorem = math.Theorem('vanderWaerdenTheorem', title='van der Waerden Theorem', contributor='van der Waerden') \
    @ r"""
    Z = U_kS_k, S_k contains {a_1, a_2, ... a_n}: arithmetic
    """

with math.Relm('Optimalization'):
    gordanLemma = math.Theorem('gordanLemma', title='Gordan Lemma',
        contributor='Gordan', applied_in=['Fritz John Theorem']) \
    @ r"""in Rn, not exists y, <Ai, y> < 0, i=1,...l, iff exists x_i>=0, x\=0, sum_i xiAi=0
    or in Rn, exists y, A' y < 0 xor A x=0 has a solution that x_i>=0, x\=0"""

    farkasLemma = math.Theorem('farkasLemma', title='Farkas Lemma',
        contributor='Farkas', applied_in=['Gordan Theorem']) \
    @ r"""in Rn, Ax=b, x>=0 has solution, xor A'y<=0, b'y>0 has solution""" \
    >> 'proved by [strict separation theorem]'
    farkasLemma.remark = 'generalized form: Motzin Theorem, Stiemke Theorem'

    fritzJohnTheorem = math.Theorem('fritzJohnTheorem', title='Fritz John Theorem', contributor='Fritz John') \
    @ r"""X* is the local opt of nlp: max_x f(x), gj(x)<=0, gj: C1 at X*, then exists aj >= 0, a \= 0 (Lagrange Multiplier)
    a0 Df(X*) - sum_j ajDgj(X*)=0, ajgj(X*)=0, aj>=0.  (Freitz John Condition)
    In trivial case when if gj is not used (j not in J), then aj=0.
    """ \
    ^ r"""X* is the local opt of nlp: max_x f(x), gj(x)<=0, gj: C1 at X*.
    {Dgj(X*), j:J} lin. indep. => Df(X*) = sum_j aj Dgj(X*), aj gj(X*)=0, aj>=0, (Karush-Kuhn-Tucker/KKT Condition)
    where ajgj(X*)=0 is known as Karush-Kuhn-Tucker complementarity condition.

    Lagrange multiplier: a
    Lagrange function: D L(x, a) = 0, where L(x,a)=f(X) - sum_j aj gj(X)
    dual problem: min_a min_x L(x, a)

    Application:
    min_x f(x), x>=0, then
    x* ⊥ Df(x*), x*, Df(x*)>=0
    """

    kuhnTuckerTheorem = math.Theorem('kuhnTuckerTheorem', title='Karush-Kuhn-Tucker Theorem', contributor='Karush, Kuhn, Tucker') \
    @ fritzJohnTheorem.remark


with math.Relm('Matrix Theory'):
    perronFrobeniusTheorem = math.Theorem('PerronFrobeniusTheorem', title='Perron Frobenius Theorem', contributor='Perron, Frobenius') \
    @ r"""A: nonneg (ie. all elements in A >=0)
    1. Ax= lambda x, lambda=rho(A)>=0, x:nonneg (lambda: Perron–Frobenius eigenvalue)
    2. inv(p-A): nonneg  <==>  p>lambda
    3. If A: indecomposable(esp. pos), lambda: pos, eigenvector x: pos, dim eigenspace = 1, lambda > sigma(A).
    3+. lambda: simple root. If A has s eigenvalues |x|=lambda(A), they are the roots x^s=lambda:
    5. if A: pos, then s=1.
    """ \
    >> '''A: pos (epsilon-room method)
    consider S={Ay>=theta y, theta>=0, y>=0 (y: unit)}
    lambda, x = argmax_y S. then Ax=lambda x.
    Or use Brouwer’s fixed point theorem
    '''
    
    perronFrobeniusTheorem.remark = 'If equation is difficult, then consider inequalities; use epsilon-room to strength the assumption'

    singularValueDecomposition = math.Theorem("SingularValueDecomposition", title="Singular Value Decomposition") \
    @ """
    A: R^mXn, rank A=r => A = USV', where U: O(m), V: O(n),
    S: diag matrix with first r diagonal elements (unique regardless of the sign).
    If A: C^mXn, then U: U(m), V: U(n).
    """ \
    >> """
    eigenvalue decomp: A'A V = VS^2, let U=AVS^{-1}
    """

    singularValueDecomposition.remark = "see Eckart-Young theorem"


with math.Relm('Graph Theory'):
    laplaceMatrix = math.Definition('LaplaceMatrix', title='Laplace Matrix')
    laplaceMatrix.content = r"""Given a graph G=<V, E>, A is adj (weight) matrix of G, D is the deg matrix of G, define
    L := D-A or L:= 1 - D^{-1/2}AD^{-1/2} (standard version)
    """


with math.Relm('Group Theory'):
    seamless = math.Theorem('Seamless', title='Seamless Lemma')
    seamless.content = r"""H<G:Grp  ==> <G-H>=G"""


with math.Relm('Category Theory'):
    yonedaLemma = math.Lemma(title='Yoneda Lemma', contributor='Yoneda', applied_in=['universal arrows',]) \
    @ 'y: N(D(r,-),K) ~ Kr, K:D->Set, r:D; y:a|-> a_r(1_r)'

    fact1 = math.Fact(title='adjunction fact', content='forall adj. yeilds u.a.')

    galoisConnectionDefinition = math.Definition(title='Galois connection') \
    @ """# (A, ≤) and (B, ≤) : POset. A monotone Galois connection between A and B consists of two monotone functions: F : A → B and G : B → A, such that forall a: A and b: B, we have
    F(a) ≤ b if and only if a ≤ G(b)."""

with math.Relm('Number Theory'):
    liouvilleNumber = math.Definition(title='Liouville Number', contributor='Liouville')\
    @ """z: irr, and for n=1,2,... exists p,q: Z satisfies
    |z-p/q| < 1/q^n
    """ \
    ^ "Liouville numbers are trans. numbers"


def query(item, section='content'):
    from fuzzywuzzy import process
    item = process.extractOne(item, math.individuals())[0]
    if section == 'all':
        print(item.content)
        if item.remark:
            print('*Remark*', item.remark)
        if item.proof:
            print('Proof.', item.proof, 'Q.E.D')
    else:
        print(getattr(item, section))

import fire
fire.Fire(query)
