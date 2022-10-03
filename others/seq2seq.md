## RNN

```
Xt , Zt-1 -> Y_t

Xt -> Zt
```



## seq2seq

seq of samples -> seq of samples



## Attention

$h_t$: hidden stata of the decoder at step/position $t$​​,

$s_k$: hidden state of the encoder
$$
a^{(t)}=\mathrm{softmax}\{s(h_t,s_k),k=1,\cdots m\}, \ln a^{(t)}\sim \{s(h_t,s_k),k=1,\cdots m\}\\
c^{(t)}=\sum_ka^{(t)}_k s_k
$$
where it is allowed to take

$s(s,h)=w s\cdot h$​​​ *scaled dot product attention*

$s(s,h)=u\cdot \tanh(Ws+Vh)$​​​​​ *Bahdanau attention*

$c_t,h_{t-1}\to h_t$​​


$$
attention(Q,K,V)=softmax(\frac{QK'}{\sqrt{d_k}}) V
$$


```python
def scaled_dot_product_attention(queries, keys, values, mask):
    # Calculate the dot product, QK_transpose
    product = tf.matmul(queries, keys, transpose_b=True)
    # Get the scale factor
    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    # Apply the scale factor to the dot product
    scaled_product = product / tf.math.sqrt(keys_dim)
    # Apply masking when it is requiered
    if mask is not None:
        scaled_product += (mask * -1e9)
    # dot product with Values
    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)
    
    return attention
```



---

**Reference**

[Attention](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634)



# GPT

