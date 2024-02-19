# Coding guidelines

## Module `forward` signature

In modules that do not implement the `Module` or ModuleT` traits, we use the
following arguments order:

- The primary input argument (e.g. hidden representations or piece
  identifiers).
- Other input arguments in alphabetical order.
- Option arguments in alphabetical order.

If there is no primary input argument, all input arguments are ordered
alphabetically.

## Module creation pattern

In Curated Transformers we often allowed layers as arguments to build models
using composition. For instance (simplified):

```python
class DecoderLayer(nn.Module):
    def __init__(self, *, attention: SelfAttention, feed_forward: PointwiseFeedForward):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
```

This works in Pytorch, because variable naming is based on attribute names.
So, in this case, the infixes `attention` and `feed_forward` are used in
variable names.

However, this does not translate well to Candle because the variable names
are constructed by pushing names to the `VarBuilder`. Since `SelfAttention`
is created separately from `DecoderLayer`, it does not get the prefix for
a transformer layer. We could push a prefix like `layer_1.attention` to the
`VarBuilder` passed to self attention, but this breaks proper abstraction,
since `DecoderLayer` should be responsible for its internal naming.

Summarized, we want `DecoderLayer` to construct its layers such as
`SelfAttention` with the correct prefixes. At the same time we want to the
layer to be generic, avoiding hardcoding the particular implementation.

To solve this, we use a creation treat similar to:

```rust
pub trait BuildAttention {
    fn build(
        &self,
        vb: VarBuilder,
    ) -> Result<Box<dyn Attention>, Box<dyn Error + Send + Sync>>;
}
```

This allows us to pass in a configuration struct to `DecoderLayer`. But
`DecoderLayer` does not have to interpret the configuration struct, since
it can construct an `Attention` module from the configuration in a generic
way.
