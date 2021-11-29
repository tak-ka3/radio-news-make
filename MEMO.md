# Memo

### tensorのsess.runの引数であるfeed_dictの使い方
```python
import tensorflow as tf
import numpy as np

a = tf.constant(3)
b = tf.placeholder(dtype=tf.int32, shape=None)
c = tf.add(a, b)

with tf.Session() as sess:
    print(sess.run(c, feed_dict={b: 1}))
    print(sess.run(c, feed_dict={b: [1, 2, 3, 4]}))
    print(sess.run(c, feed_dict={b: np.random.randint(0, 9, 10)}))
    
# 以下，実行結果
# 4
# [4 5 6 7]
# [ 4  7  8  4  7  3  4  9 10  4]
```

### 複数の文入力を可能にする
- outputにはsample_sequenceという関数を渡しているので、この関数をいじれば複数入力が行けそう。特に、引数contextをいじるのが良さそう。
