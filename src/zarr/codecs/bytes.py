from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import Buffer, NDArrayLike, NDBuffer
from zarr.core.common import JSON, parse_enum, parse_named_configuration
from zarr.core.dtype.common import HasEndianness

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec


class Endian(Enum):
    """
    Enum for endian type used by bytes codec.
    """

    big = "big"
    little = "little"


default_system_endian = Endian(sys.byteorder)


@dataclass(frozen=True)
class BytesCodec(ArrayBytesCodec):
    """
    Bytes codec for converting arrays to/from raw bytes.

    This codec handles the conversion between typed NumPy arrays and raw byte
    representations. It's typically the first codec in a Zarr v3 codec pipeline,
    converting array data to bytes before applying compression or other transformations.

    Attributes
    ----------
    is_fixed_size : bool
        Always True, as the byte representation has a predictable size based on
        the array shape and dtype.
    endian : Endian or None
        The byte order (endianness) to use for multi-byte data types.

    Parameters
    ----------
    endian : {'big', 'little'} or Endian or None, optional
        The byte order for multi-byte data types:

        - 'big': Big-endian byte order (most significant byte first)
        - 'little': Little-endian byte order (least significant byte first)
        - None: No byte order specified (only valid for single-byte types)

        Default: system native byte order.

    Examples
    --------
    Create a bytes codec with system endianness:

    >>> from zarr.codecs import BytesCodec
    >>> codec = BytesCodec()

    Create with explicit little-endian:

    >>> codec = BytesCodec(endian='little')
    >>> codec.endian
    <Endian.little: 'little'>

    Use in a codec pipeline:

    >>> import zarr
    >>> from zarr.codecs import BytesCodec, ZstdCodec
    >>> arr = zarr.create(
    ...     shape=(100, 100),
    ...     chunks=(10, 10),
    ...     dtype='f8',
    ...     zarr_format=3,
    ...     codecs=[BytesCodec(), ZstdCodec()]
    ... )

    Notes
    -----
    In Zarr v3, the BytesCodec is required as the array-to-bytes codec in the
    codec pipeline. It must appear before any bytes-to-bytes codecs (like
    compression codecs).

    For single-byte data types (like 'uint8' or 'int8'), the endianness parameter
    is not required and will be set to None during array specification evolution.

    See Also
    --------
    Endian : Enum for byte order options
    ZstdCodec : Compression codec often used after BytesCodec
    BloscCodec : Alternative compression codec
    """

    is_fixed_size = True

    endian: Endian | None

    def __init__(self, *, endian: Endian | str | None = default_system_endian) -> None:
        endian_parsed = None if endian is None else parse_enum(endian, Endian)

        object.__setattr__(self, "endian", endian_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "bytes", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        if self.endian is None:
            return {"name": "bytes"}
        else:
            return {"name": "bytes", "configuration": {"endian": self.endian.value}}

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if not isinstance(array_spec.dtype, HasEndianness):
            if self.endian is not None:
                return replace(self, endian=None)
        elif self.endian is None:
            raise ValueError(
                "The `endian` configuration needs to be specified for multi-byte data types."
            )
        return self

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        assert isinstance(chunk_bytes, Buffer)
        # TODO: remove endianness enum in favor of literal union
        endian_str = self.endian.value if self.endian is not None else None
        if isinstance(chunk_spec.dtype, HasEndianness):
            dtype = replace(chunk_spec.dtype, endianness=endian_str).to_native_dtype()  # type: ignore[call-arg]
        else:
            dtype = chunk_spec.dtype.to_native_dtype()
        as_array_like = chunk_bytes.as_array_like()
        if isinstance(as_array_like, NDArrayLike):
            as_nd_array_like = as_array_like
        else:
            as_nd_array_like = np.asanyarray(as_array_like)
        chunk_array = chunk_spec.prototype.nd_buffer.from_ndarray_like(
            as_nd_array_like.view(dtype=dtype)
        )

        # ensure correct chunk shape
        if chunk_array.shape != chunk_spec.shape:
            chunk_array = chunk_array.reshape(
                chunk_spec.shape,
            )
        return chunk_array

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        assert isinstance(chunk_array, NDBuffer)
        if (
            chunk_array.dtype.itemsize > 1
            and self.endian is not None
            and self.endian != chunk_array.byteorder
        ):
            # type-ignore is a numpy bug
            # see https://github.com/numpy/numpy/issues/26473
            new_dtype = chunk_array.dtype.newbyteorder(self.endian.name)  # type: ignore[arg-type]
            chunk_array = chunk_array.astype(new_dtype)

        nd_array = chunk_array.as_ndarray_like()
        # Flatten the nd-array (only copy if needed) and reinterpret as bytes
        nd_array = nd_array.ravel().view(dtype="B")
        return chunk_spec.prototype.buffer.from_array_like(nd_array)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length
