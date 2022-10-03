# NLP

[TOC]

## TF-IDF

- $D$: documents
- $D_i$: documents containing word $i$
- $tf_{ij}$: freq of word $i$ in document $j$

$$
tf_{ij}=\frac{n_{ij}}{\sum_kn_{kj}}\\
idf_{i}=\log\frac{|D|}{1+|D_i|}\\
tf-idf_{ij}=tf_{ij}idf_i
$$

```python
from gensim import corpora, models
dictionary = corpora.Dictionary(docs)
bow = [dictionary.doc2bow(doc) for doc in docs]  #tf
tfidf=models.TfidfModel(bow)  # tf-idf
bow=tfidf[bow]
```



## TextRank

$$
WS(v)=(1-d)+d\sum_{u\to v}\frac{w_{uv}}{\sum_{u\to x}w_{ux}}WS(u),
$$

$d$: damping coef.

## BOW(bag of words)

word freq. in a document

```python
from nltk.probability import FreqDist
fd = FreqDist(words)
```



## word2vec

### History

T. Mikolov(2013)

### Model

$C(w)$

network model:
$$
y=b+Wx+U\tanh(d+Hx)\\
P(w_t=i|context)=softmax(y) \sim C(w_t)
$$
where $x=C(w_{t-n+1})\cdots C(w_{t-1}), t=C(w_t)$.

```python
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname # pathlib.Path

    def __iter__(self):
        for fname in self.dirname.iterdir():
            for line in open(fname):
                yield line.split()

sentences = MySentences('/some/directory')
model = gensim.models.Word2Vec(sentences, iter=1)

model = gensim.models.Word2Vec(iter=1)  # an empty model, no training yet
model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
model.train(other_sentences)
```

### skip-gram word2vec

$P((X_1,X_3), X_2)$

$X_2$ pred $X_1,X_3$

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.preprocessing.text import *
from keras.preprocessing.sequence import skipgrams

text='I love green eggs and ham'
tokenizer=Tokenizer()
tokenizer.fit_on_texts([text])

def text_to_word_seqence(text):
    return text.lower().split()

# word <-> id
word2id=tokenizer.word_index
id2word={v:k for k, v in word2id.items()}

wids=[word2id[w] for w in text_to_word_seqence(text)]

pairs, labels = skipgrams(wids, len(word2id))
for p, l in zip(pairs, labels):
    print(p, l)  # (love i) 1
```



## Topic/Keywords Model

*Model:* suppose $w_i\perp d_j$ rwt $t_k$
$$
p(w_i|d_j)=\sum_kp(w_i|t_k)p(t_k|d_j)\\
p(w_i, d_j)=\sum_kp(w_i|t_k)p(d_j|t_k)p(t_k)
$$
*fitting(decomp.)*: input $N(w_i,d_j)$ or $p(w_i|d_j)$ output $p(w_i|t_k),p(t_k|d_j)$

*predict*: input $N(w_i,d)$ or $p(w_i|d)$, output $p(t_k,d)$

*word matrix* $p(w_i|d_j)$

*joint word matrix* $p(w_i, d_j)$
$$
P=W\Lambda D^T, s.m. decomp.
$$


### LSA/LSI

BOW+SVD(PCA)

words-documents: $N\times p$

bow $A: N\times p$

topic $C: N\times k$

rep. of documents $A=CV^T$

```python
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=k)
topic=lsi[bow]
```



### pLSA

Model (4)

**Algorithm(T. Hofmann)*

- initial-step

  $p(z),p(d|z),p(w|z)$

- E-step

  $P(z|d,w)=\frac{p(z)p(d|z)p(w|z)}{p(d,w)}$

- M-step
  $$
  p(w|z)\sim \int_D n(d,w)p(z|d,w)\\
  p(d|z)\sim \int_W n(d,w)p(z|d,w)\\
  p(z)\sim \int_{D,W} n(d,w)p(z|d,w)
  $$
  i.e. $\max  L=\int_Z(\log p'(w|z)\int_D n(d,w)p(z|d,w))$...

*See also* NMF

### LDA



### Model

*Notations*: $z$: topic, $w$: word, $d=(w, w, \cdots,w)$: document/sentence (word seq.)

*Model*

1. prior dist.

$\theta, \phi\sim Dir$

2. cond dist.

$Z|\theta \sim Multi(\theta)$, 

$W|Z,\phi\sim Multi(\phi_Z)$, in $Z$, the dist. of $W$

3. hypo.

$P(w_i|d_j)=\sum_kP(w_i|z_k)P(z_k|d_j)$  $w_i\perp d_j | z_k$

 

$P(z_i|z_{-i}, w) \propto (n_{i,-i}+\beta_i)/(\sum_in_{i,-i}+\beta_i)(n_{i,-i}+\alpha_i) $

$n_{i,-i}=\sharp\{z_i|z_{-i}\}$



Samples: $D_j=\{W_{ij}\}$

Estimate
$$
\hat{\phi}_{si}=(n_{si}+\beta_i)/(\sum_iN_{i|k}+\beta_i)\\
\hat{\phi}_{js}=(n_{js}+\beta_i)/(\sum_iN_{k|j}+\beta_i)
$$


### Example

Samples of sentences:

- I like to eat broccoli and bananas.
- I ate a banana and spinach smoothie for breakfast.
- Chinchillas and kittens are cute.
- My sister adopted a kitten yesterday.
- Look at this cute hamster munching on a piece of broccoli.

$d_1,d_2$: 100% $z_1$
$d_3,d_4$: 100% $z_2$
$d_5$: 60% Topic A, 40% $z_3$

$z_1$: 30% broccoli, 15% bananas, 10% breakfast, 10% munching, … (about food)
$z_2 $: 20% chinchillas, 20% kittens, 20% cute, 15% hamster, … (about cute animals)

```python
lda=models.LdaModel(bow, id2word=dictionary, num_topics=k)
lda[bow]
```

### References

[LDA wiki](<https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>)

Blei(2003)

[LDA](<https://www.jianshu.com/p/24b1bca1629f>)



## Similarity of text

### Model

Text: $a, b$

Word: $a_i,b_j$

Similarity Matrix:

$A=\{sim(a_i, b_j)\}_{ij},B=\{sim(b_i, a_j)\}_{ij}$,

$sim(x,y):=MLP(x)\cdot MLP(y)$



Stucture:

Layer-1

$b'=Ab, a'=Ba $

Layer-2

$v_a=MLP(a,b'),v_b=MLP(b,a')$

$v_a,v_b$=> MLP-classifier



### Ref.

Parikh2016



## n-Grammar

### Model

text: $w=(w_1,\cdots,w_n),w_i\in \R^d$;$w\in \R^{nd},w\in \R^{d\times n}$

window: $k$, $x\in \R^{kd},x\in \R^{d\times k}$

$x$=> Conv => Pooling

### Ref.

Kalchbrenner(2014)

Johnson-Zhang(2015)

Le-Zuidema(2015): dep-tree



## RNN

### RNN

$y_n=RNN(x)=R(\cdots R(s_0,x_1), \cdots, x_n)$

$y^*=RNN^*(x)=(y_1,\cdots, y_n)$

### biRNN

$biRNN(x):=(RNN(x_{1:i}), RNN(x_{n:i}))$

**Ref.**

Irsoy, Cardie(2014)

### deep RNN



**Ref.**

Hihi, Benjio(1996)

Sutskever(2014)

### stack RNN

**Ref.**

Dyer(2015), Watanabe, Sumita(2015)



### SRNN(Elman RNN)

Mikolov(2012)



### LSTM

Hochreiter, Schmidhuber(97)

### GRU

Cho(2014)