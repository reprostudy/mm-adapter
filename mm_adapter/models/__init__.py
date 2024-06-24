from enum import Enum
from functools import partial

from mm_adapter.models._base_model import BaseModel
from mm_adapter.models.clip_classifier import ClipClassifier


class Model(Enum):
    CLIP_VIT_BASE_PATCH16_PRETRAINED = partial(lambda: ClipClassifier("ViT-B/16"))

    @classmethod
    def from_str(cls, name: str) -> BaseModel:
        model: BaseModel = cls[name.upper()].value()
        return model
