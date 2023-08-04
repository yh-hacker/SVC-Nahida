import copy
import random
from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as t_func
#from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


class Hubert(paddle.nn.Layer):
    def __init__(self, num_label_embeddings: int = 100, mask: bool = True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(
            nn.TransformerEncoderLayer(
                768, 12, 3072, activation="gelu"
            ),
            12,
        )
        self.proj = nn.Linear(768, 256)

        self.masked_spec_embed = paddle.create_parameter([768],dtype = 'float32')
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, None, 2)
            x[mask] = self.masked_spec_embed
        return x, mask

    def encode(
            self, x: paddle.Tensor, layer: Optional[int] = None
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose([0, 2, 1]))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: paddle.Tensor) -> paddle.Tensor:
        logits = t_func.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            axis=-1,
        )
        return logits / 0.1

    def forward(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        x, mask = self.encode(x)
        x = self.proj(x)
        logits = self.logits(x)
        return logits, mask


class HubertSoft(Hubert):
    def __init__(self):
        super().__init__()

    def units(self, wav: paddle.Tensor) -> paddle.Tensor:
        wav = t_func.pad(wav, ((400 - 320) // 2, (400 - 320) // 2),data_format='NCL')
        x, _ = self.encode(wav)
        return self.proj(x)


class FeatureExtractor(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1D(1, 512, 10, 5, bias_attr=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1D(512, 512, 3, 2, bias_attr=False)
        self.conv2 = nn.Conv1D(512, 512, 3, 2, bias_attr=False)
        self.conv3 = nn.Conv1D(512, 512, 3, 2, bias_attr=False)
        self.conv4 = nn.Conv1D(512, 512, 3, 2, bias_attr=False)
        self.conv5 = nn.Conv1D(512, 512, 2, 2, bias_attr=False)
        self.conv6 = nn.Conv1D(512, 512, 2, 2, bias_attr=False)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = t_func.gelu(self.norm0(self.conv0(x)))
        x = t_func.gelu(self.conv1(x))
        x = t_func.gelu(self.conv2(x))
        x = t_func.gelu(self.conv3(x))
        x = t_func.gelu(self.conv4(x))
        x = t_func.gelu(self.conv5(x))
        x = t_func.gelu(self.conv6(x))
        return x


class FeatureProjection(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1D(
            768,
            768,
            kernel_size=128,
            padding=128 // 2,
            groups=16,
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x.transpose([0, 2, 1]))
        x = t_func.gelu(x[:, :, :-1])
        return x.transpose([0, 2, 1])


class TransformerEncoder(paddle.nn.Layer):
    def __init__(
            self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.LayerList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
            self,
            src: paddle.Tensor,
            mask: paddle.Tensor = None,
            src_key_padding_mask: paddle.Tensor = None,
            output_layer: Optional[int] = None,
    ) -> paddle.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
        return output


def _compute_mask(
        shape: Tuple[int, int],
        mask_prob: float,
        mask_length: int,
        device: None,
        min_masks: int = 0,
) -> paddle.Tensor:
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    mask = paddle.zeros((batch_size, sequence_length), dtype='bool')

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = paddle.ones(
        (batch_size, sequence_length - (mask_length - 1))
    )
    
    # get random indices to mask
    mask_indices = paddle.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    mask_indices = (
        mask_indices.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        paddle.arange(mask_length)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    mask_idxs = mask_indices + offsets

    # scatter indices to mask
    mask = mask.scatter(1, mask_idxs, True)

    return mask


def hubert_soft(
        path: str,
) -> HubertSoft:
    r"""HuBERT-Soft from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        path (str): path of a pretrained model
    """
    hubert = HubertSoft()
    checkpoint = paddle.load(path)
    #consume_prefix_in_state_dict_if_present(checkpoint, "module.")
    hubert.set_state_dict(checkpoint)
    hubert.eval()
    return hubert

if __name__ == '__main__':
    hubert = HubertSoft()
    d = paddle.load(r'E:\trans\hubert\final.pdparams')
    hubert.set_state_dict(d)
