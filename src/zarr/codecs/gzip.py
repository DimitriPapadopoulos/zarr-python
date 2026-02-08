from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from numcodecs.gzip import GZip

from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


def parse_gzip_level(data: JSON) -> int:
    if not isinstance(data, (int)):
        raise TypeError(f"Expected int, got {type(data)}")
    if data not in range(10):
        raise ValueError(
            f"Expected an integer from the inclusive range (0, 9). Got {data} instead."
        )
    return data


@dataclass(frozen=True)
class GzipCodec(BytesBytesCodec):
    """
    Gzip compression codec for zarr.

    Gzip is a widely-used lossless compression format based on the DEFLATE algorithm.
    It provides good compression ratios with reasonable speed, making it suitable for
    general-purpose data compression.

    Attributes
    ----------
    is_fixed_size : bool
        Always False for Gzip codec, as compression produces variable-sized output.
    level : int
        The compression level (0-9).

    Parameters
    ----------
    level : int, optional
        The compression level, from 0 (no compression) to 9 (maximum compression).
        Higher values provide better compression at the cost of speed. Default: 5.

    Examples
    --------
    Create a Gzip codec with default compression:

    >>> from zarr.codecs import GzipCodec
    >>> codec = GzipCodec()
    >>> codec.level
    5

    Create with maximum compression:

    >>> codec = GzipCodec(level=9)
    >>> codec.level
    9

    Use in array creation:

    >>> import zarr
    >>> from zarr.codecs import BytesCodec, GzipCodec
    >>> arr = zarr.create(
    ...     shape=(100, 100),
    ...     chunks=(10, 10),
    ...     dtype='f8',
    ...     zarr_format=3,
    ...     codecs=[BytesCodec(), GzipCodec(level=6)]
    ... )

    Notes
    -----
    Gzip compression is slower than algorithms like LZ4 but often achieves better
    compression ratios. It's a good choice when storage size is more important
    than compression/decompression speed.

    See Also
    --------
    BloscCodec : High-performance compression codec with multiple algorithms
    ZstdCodec : Modern compression codec with good balance of speed and ratio
    """

    is_fixed_size = False

    level: int = 5

    def __init__(self, *, level: int = 5) -> None:
        level_parsed = parse_gzip_level(level)

        object.__setattr__(self, "level", level_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "gzip")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "gzip", "configuration": {"level": self.level}}

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, GZip(self.level).decode, chunk_bytes, chunk_spec.prototype
        )

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, GZip(self.level).encode, chunk_bytes, chunk_spec.prototype
        )

    def compute_encoded_size(
        self,
        _input_byte_length: int,
        _chunk_spec: ArraySpec,
    ) -> int:
        raise NotImplementedError
