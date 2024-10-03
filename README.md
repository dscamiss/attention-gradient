# attention-gradient

A PyTorch implementation of scaled dot-product attention,
with an explicit `backward()` method to compute gradients.

The implementation follows this blog post: https://dscamiss.github.io/blog/posts/attention-explicit/.

# Usage

```shell
git clone https://github.com/dscamiss/attention-gradient && cd attention-gradient
pip install -r requirements.txt
python -m pytest .
```