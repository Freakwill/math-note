# Hindman thm

J. E. Baumgartner(1974)

## Background

### Hindman thm

$\mathbb{N}=\bigcup_{i=1}^kA_i, \exists A_i, A_i$ is closed under $+$.

### Hindman thm 2

$\mathcal{F}:=\mathcal{P}_\omega(\mathbb{N})$. $\mathcal{F}=\bigcup_{i=1}^k\mathcal{A}_i, \exists \mathcal{A}_i\supset \mathcal{D}$ (inf. disjoint) and $FU(\mathcal{D})\subset\mathcal{A}_i$.


### Def

* $FU(\mathcal{D})=$ all finite (not empty) unions of elms of $D$,$\mathcal{D}\subset \mathcal{F}$
* $\mathcal{X}\subset \mathcal{F}$ is large for $\mathcal{D}$ (infinte disjoint) iff for all (inf. disjoint) $\mathcal{D}'\subset FU(\mathcal{D}), FU(\mathcal{D}')\cap \mathcal{X}\neq\emptyset$. (otherwise, exists (inf. disjoint) $\mathcal{D}'\subset FU(\mathcal{D}), FU(\mathcal{D}')\cap \mathcal{X}=\emptyset$)

*Remark.* $FU$ is a closure op. $\mathcal{X}\subset \mathcal{F}$ is large for $\mathcal{D}$ then $\mathcal{X}$ is inf. and is large after deleting some finite elms.


### Lemma 1
* If $\mathcal{X}$ is large for $\mathcal{D}$, $\mathcal{X}=\mathcal{Y}\cup \mathcal{Z}$, then $\exists \mathcal{D}'\subset FU(\mathcal{D})$ (inf. disjoint), $\mathcal{Y}$ or $\mathcal{Z}$ is large for $\mathcal{D}'$

* If $\mathcal{X}$ is large for $\forall n\geq0, \{X\in\mathcal{X},\min X>n\}$ is large for $\mathcal{D}$.


### Lemma 2
Suppose $\mathcal{X}$ is large for $\mathcal{D}$. $\exists \mathcal{E} (fin.) \subset FU(\mathcal{D})\forall X\in FU(\mathcal{D})$, if $X\cap\bigcup \mathcal{E}=\emptyset$ then $\exists D\in FU(\mathcal{E}), X\cup D\in \mathcal{X}$. (prompt: otherwise (inf. disjoint) $\exists X_k\in FU(\mathcal{D}), X_k\cup D\not\in \mathcal{X}, D=\bigcup_{i<k}X_k$.)


### Lemma 3
Suppose $\mathcal{X}$ is large for $\mathcal{D}$. $\exists D\in FU(\mathcal{D}), \{X\in \mathcal{X},X\cup D\in \mathcal{X}\}$ is large for some $\mathcal{D}'\subset FU(\mathcal{D})$.

*Proof.* Let $\mathcal{E}$ be as in Lemma 2 and $\mathcal{D}_1$ is an inf. disjoint collection that $\forall X\in FU(\mathcal{D}_1), X\cap\bigcup \mathcal{E}=\emptyset$. ($(\mathcal{E}, \mathcal{D}_1)$: Baumgartner decomp.) 

$$\mathcal{X}\cap \mathcal{D}_1\subset \bigcup\{\mathcal{X}_D, D\in FU(\mathcal{E})\},$$

where $\mathcal{X}_D=\{X\in \mathcal{X}, X\cup D\in \mathcal{X}\}$. Then apply Lemma 1.


### Lemma 4 (Baumgartner Lemma)
If $\mathcal{X}$ is large for $\mathcal{D}$, then $\mathcal{D}'\subset FU(\mathcal{D})$ that $FU(\mathcal{D'})\subset \mathcal{X}$.


*Proof.* Construct Baumgartner seq.