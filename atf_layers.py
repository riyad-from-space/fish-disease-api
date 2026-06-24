"""
Custom Keras layers for the ATF-Net (Attention Cross Fusion) model.

The trained model `ATF_Net_Fusion_Model.h5` was built with these 8 custom
layer classes defined inline in the training notebook. A .h5 file stores only
the *weights* and *config* (hyper-parameters) of custom layers — never their
Python `call()` logic — so the original class definitions MUST be available at
load time, passed via `custom_objects=` (or, as here, registered globally with
`@keras.saving.register_keras_serializable`).

These classes were reconstructed from the saved weight manifest (variable names,
shapes and ordering) + the model connectivity graph, so that:
  * `build()` creates internal sub-layers with weight shapes/order that exactly
    match the saved file (otherwise weight loading fails or mismaps), and
  * `call()` implements the standard, canonical math for each component
    (ViT-Tiny encoder, triple cross-attention fusion, feature projection).

Architecture recap (input 224x224x3):
    input -> [Custom_CNN(->256), ResNet50(->256), ViT-Tiny(->192)]
          -> 3x FeatureProjection(->256)
          -> concat(768) -> dense_88(768) ─┐
          -> TripleCrossAttention(->768) ──┴─ add -> 512 -> 256 -> 128 -> 7
"""

import keras
from keras import layers, ops


# ---------------------------------------------------------------------------
# Vision Transformer (ViT-Tiny) building blocks
# embed_dim=192, num_heads=3, depth=12, patch=16  (224/16 -> 14x14 = 196 patches)
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="ATFNet")
class PatchEmbedding(layers.Layer):
    """Split the image into non-overlapping 16x16 patches, flatten each to
    16*16*3=768 values, and linearly project to `embed_dim` with a Dense layer
    (saved weight: patch_embed/dense_16/kernel (768, 192))."""

    def __init__(self, patch_size=16, embed_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Patch embedding is a *linear* projection (no activation) — standard ViT.
        self.proj = layers.Dense(embed_dim, name="dense_16")

    def build(self, input_shape):
        patch_dim = self.patch_size * self.patch_size * 3
        self.proj.build((None, None, patch_dim))
        super().build(input_shape)

    def call(self, x):
        patches = ops.image.extract_patches(
            x, size=self.patch_size, strides=self.patch_size, padding="valid"
        )  # -> (B, 14, 14, 768)
        b = ops.shape(patches)[0]
        patch_dim = self.patch_size * self.patch_size * 3
        patches = ops.reshape(patches, (b, -1, patch_dim))  # (B, 196, 768)
        return self.proj(patches)  # (B, 196, 192)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(patch_size=self.patch_size, embed_dim=self.embed_dim)
        return cfg


@keras.saving.register_keras_serializable(package="ATFNet")
class PrependClassToken(layers.Layer):
    """Prepend a learnable [CLS] token to the patch sequence.
    Saved weight: cls_token/cls_token (1, 1, 192) -> 196 patches become 197 tokens."""

    def __init__(self, embed_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, self.embed_dim),
            initializer="zeros", trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        b = ops.shape(x)[0]
        cls = ops.broadcast_to(self.cls_token, (b, 1, self.embed_dim))
        cls = ops.cast(cls, x.dtype)
        return ops.concatenate([cls, x], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(embed_dim=self.embed_dim)
        return cfg


@keras.saving.register_keras_serializable(package="ATFNet")
class AddPositionalEmbedding(layers.Layer):
    """Add a learnable positional embedding to the (197-token) sequence.
    Saved weight: pos_embed/pos_embed (1, 197, 192)."""

    def __init__(self, num_tokens=197, embed_dim=192, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed", shape=(1, self.num_tokens, self.embed_dim),
            initializer="zeros", trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + ops.cast(self.pos_embed, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(num_tokens=self.num_tokens, embed_dim=self.embed_dim)
        return cfg


@keras.saving.register_keras_serializable(package="ATFNet")
class TransformerBlock(layers.Layer):
    """Pre-norm Transformer encoder block (canonical ViT).
    Saved weight order per block:
        layer_normalization (ln1) -> multi_head_attention -> layer_normalization_1 (ln2)
        -> dense_17 (MLP 192->768) -> dense_18 (MLP 768->192)
    => sub-layers are created in exactly that order so weights map 1:1.
    embed_dim=192, num_heads=3, key_dim=64, mlp_dim=768, GELU MLP activation."""

    def __init__(self, embed_dim=192, num_heads=3, mlp_dim=768, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name="layer_normalization")
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            dropout=dropout, name="multi_head_attention",
        )
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name="layer_normalization_1")
        self.mlp1 = layers.Dense(mlp_dim, activation="gelu", name="dense_17")
        self.mlp2 = layers.Dense(embed_dim, name="dense_18")

    def build(self, input_shape):
        self.ln1.build(input_shape)
        self.mha.build(input_shape, input_shape)
        self.ln2.build(input_shape)
        self.mlp1.build(input_shape)
        mlp_mid = tuple(input_shape[:-1]) + (self.mlp_dim,)
        self.mlp2.build(mlp_mid)
        super().build(input_shape)

    def call(self, x, training=None):
        h = self.ln1(x)
        x = x + self.mha(h, h, training=training)
        h = self.ln2(x)
        x = x + self.mlp2(self.mlp1(h))
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(embed_dim=self.embed_dim, num_heads=self.num_heads,
                   mlp_dim=self.mlp_dim, dropout=self.dropout)
        return cfg


@keras.saving.register_keras_serializable(package="ATFNet")
class ExtractToken(layers.Layer):
    """Return a single token from the sequence (the [CLS] token at index 0)."""

    def __init__(self, index=0, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update(index=self.index)
        return cfg


# ---------------------------------------------------------------------------
# Fusion head building blocks
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="ATFNet")
class FeatureProjection(layers.Layer):
    """Project a backbone feature vector to a common `dim` (=256) space.
    Saved weight order: dense_85 (Dense, ->256) -> batch_normalization_658 (BN).
    Dense -> BatchNorm -> Dropout, with NO activation (linear projection).

    The "no activation" was confirmed empirically: a ReLU here collapses the
    model's confidence (mean ~0.44 vs ~0.78 linear) and lowers healthy/diseased
    accuracy on a labelled test set — i.e. the trained weights only behave
    correctly when this projection is linear, which is also the conventional
    design for a feature-projection layer feeding a residual fusion."""

    def __init__(self, dim=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dropout = dropout
        self.dense = layers.Dense(dim, name="dense")
        self.bn = layers.BatchNormalization(name="batch_normalization")
        self.drop = layers.Dropout(dropout)

    def build(self, input_shape):
        self.dense.build(input_shape)
        projected = tuple(input_shape[:-1]) + (self.dim,)
        self.bn.build(projected)
        super().build(input_shape)

    def call(self, x, training=None):
        x = self.dense(x)
        x = self.bn(x, training=training)
        x = self.drop(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dim=self.dim, dropout=self.dropout)
        return cfg


@keras.saving.register_keras_serializable(package="ATFNet")
class ConcatLayer(layers.Layer):
    """Concatenate the three projected feature vectors along the feature axis
    (3 x 256 -> 768). Carries no weights."""

    def call(self, inputs):
        return ops.concatenate(inputs, axis=-1)


@keras.saving.register_keras_serializable(package="ATFNet")
class TripleCrossAttention(layers.Layer):
    """Cross-modal fusion of the three projected streams.

    Saved weights: 3 independent MultiHeadAttention blocks
    (num_heads=4, key_dim=256). One MHA per stream: each stream (as a query)
    attends over the stacked set of all three streams (context), and the three
    attention outputs are concatenated back to 768-d — matching the residual
    `add` with the 768-d linear fusion (dense_88) downstream.
    """

    def __init__(self, dim=256, num_heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        # key_dim == dim (saved query kernel is (256, 4, 256)).
        self.mha_a = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim, dropout=dropout, name="multi_head_attention_a")
        self.mha_b = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim, dropout=dropout, name="multi_head_attention_b")
        self.mha_c = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim, dropout=dropout, name="multi_head_attention_c")

    def build(self, input_shape):
        # input_shape is a list of 3 shapes, each (B, dim)
        q_shape = (input_shape[0][0], 1, self.dim)
        kv_shape = (input_shape[0][0], 3, self.dim)
        for mha in (self.mha_a, self.mha_b, self.mha_c):
            mha.build(q_shape, kv_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        a, b, c = inputs
        a1 = ops.expand_dims(a, axis=1)  # (B,1,dim)
        b1 = ops.expand_dims(b, axis=1)
        c1 = ops.expand_dims(c, axis=1)
        context = ops.concatenate([a1, b1, c1], axis=1)  # (B,3,dim)
        out_a = self.mha_a(a1, context, training=training)[:, 0, :]
        out_b = self.mha_b(b1, context, training=training)[:, 0, :]
        out_c = self.mha_c(c1, context, training=training)[:, 0, :]
        return ops.concatenate([out_a, out_b, out_c], axis=-1)  # (B, 3*dim=768)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dim=self.dim, num_heads=self.num_heads, dropout=self.dropout)
        return cfg


# Convenience dict for keras.models.load_model(..., custom_objects=ATF_CUSTOM_OBJECTS)
ATF_CUSTOM_OBJECTS = {
    "PatchEmbedding": PatchEmbedding,
    "PrependClassToken": PrependClassToken,
    "AddPositionalEmbedding": AddPositionalEmbedding,
    "TransformerBlock": TransformerBlock,
    "ExtractToken": ExtractToken,
    "FeatureProjection": FeatureProjection,
    "ConcatLayer": ConcatLayer,
    "TripleCrossAttention": TripleCrossAttention,
}
