from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class EncodedSample:
    data: bytes
    codec: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError("EncodedSample.data must be bytes")


class CodecAdapter(ABC):
    name: str

    @abstractmethod
    def encode(self, image: Image.Image) -> EncodedSample:
        raise NotImplementedError

    @abstractmethod
    def decode(self, data: bytes) -> Image.Image:
        raise NotImplementedError
