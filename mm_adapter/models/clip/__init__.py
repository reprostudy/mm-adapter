from mm_adapter.models.clip.clip_adapter import CLIPAdapter
from mm_adapter.models.clip.clip_base import ClipBase
from mm_adapter.models.clip.clip_linear import ClipLinear
from mm_adapter.models.clip.clip_mlp_head import CLIPMLPHead
from mm_adapter.models.clip.clip_mm_mlp_adapter import CLIPMMMLPAdapter
from mm_adapter.models.clip.clip_transformer import ClipTransformer
from mm_adapter.models.clip.clip_transformer_adapter import CLIPTransformerAdapter
from mm_adapter.models.clip.clip_transformer_adapter_text import CLIPTransformerAdapterText
from mm_adapter.models.clip.clip_transformer_downscaled import ClipTransformerDownscaled
from mm_adapter.models.clip.clip_transformer_w_hypernet import ClipTransformerWHypernet

MODELS = {
    "clip_base": ClipBase,
    "clip_linear": ClipLinear,
    "clip_transformer": ClipTransformer,
    "clip_transformer_downscaled": ClipTransformerDownscaled,
    "clip_mm_mlp": CLIPMLPHead,
    "clip_mm_mlp_adapter": CLIPMMMLPAdapter,
    "clip_transformer_w_hypernet": ClipTransformerWHypernet,
    "clip_transformer_adapter": CLIPTransformerAdapter,
    "clip_transformer_adapter_text": CLIPTransformerAdapterText,
    "clip_adapter": CLIPAdapter,
}
