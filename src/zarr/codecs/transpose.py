from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np

from zarr.abc.codec import ArrayArrayCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.common import JSON, parse_named_configuration

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.buffer import NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


def parse_transpose_order(data: JSON | Iterable[int]) -> tuple[int, ...]:
    if not isinstance(data, Iterable):
        raise TypeError(f"Expected an iterable. Got {data} instead.")
    if not all(isinstance(a, int) for a in data):
        raise TypeError(f"Expected an iterable of integers. Got {data} instead.")
    return tuple(cast("Iterable[int]", data))


@dataclass(frozen=True)
class TransposeCodec(ArrayArrayCodec):
    """
    Transpose codec for reordering array dimensions.

    This codec performs dimension transposition on array chunks before they are
    stored. It can be useful for optimizing access patterns or changing the memory
    layout of chunked data.

    Attributes
    ----------
    is_fixed_size : bool
        Always True, as transposition doesn't change the total number of elements.
    order : tuple of int
        The permutation of dimensions to apply.

    Parameters
    ----------
    order : iterable of int
        A permutation of the dimension indices. For an N-dimensional array, this
        should be a tuple containing each integer from 0 to N-1 exactly once.
        For example, (2, 0, 1) transposes a 3-D array by moving the third dimension
        to the first position, the first to the second, and the second to the third.

    Examples
    --------
    Transpose a 2-D array (swap dimensions):

    >>> from zarr.codecs import TransposeCodec
    >>> codec = TransposeCodec(order=(1, 0))
    >>> codec.order
    (1, 0)

    Transpose a 3-D array:

    >>> codec = TransposeCodec(order=(2, 0, 1))
    >>> codec.order
    (2, 0, 1)

    Use in array creation:

    >>> import zarr
    >>> from zarr.codecs import TransposeCodec, BytesCodec, ZstdCodec
    >>> # Create array with transposed storage
    >>> arr = zarr.create(
    ...     shape=(100, 200, 300),
    ...     chunks=(10, 20, 30),
    ...     dtype='f4',
    ...     zarr_format=3,
    ...     codecs=[TransposeCodec(order=(2, 1, 0)), BytesCodec(), ZstdCodec()]
    ... )

    Notes
    -----
    The transpose codec is an array-to-array codec and must appear before the
    array-to-bytes codec (typically BytesCodec) in the codec pipeline.

    Transposition can improve access performance when the natural access pattern
    doesn't match the storage order. For example, if data is stored in row-major
    order but typically accessed column-wise, transposition can optimize storage.

    The order tuple must:
    - Have the same length as the number of array dimensions
    - Contain each dimension index exactly once
    - Use indices from 0 to ndim-1

    See Also
    --------
    BytesCodec : Array-to-bytes codec typically used after TransposeCodec
    """

    is_fixed_size = True

    order: tuple[int, ...]

    def __init__(self, *, order: Iterable[int]) -> None:
        order_parsed = parse_transpose_order(order)

        object.__setattr__(self, "order", order_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "transpose")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "transpose", "configuration": {"order": tuple(self.order)}}

    def validate(
        self,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        if len(self.order) != len(shape):
            raise ValueError(
                f"The `order` tuple must have as many entries as there are dimensions in the array. Got {self.order}."
            )
        if len(self.order) != len(set(self.order)):
            raise ValueError(
                f"There must not be duplicates in the `order` tuple. Got {self.order}."
            )
        if not all(0 <= x < len(shape) for x in self.order):
            raise ValueError(
                f"All entries in the `order` tuple must be between 0 and the number of dimensions in the array. Got {self.order}."
            )

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        ndim = array_spec.ndim
        if len(self.order) != ndim:
            raise ValueError(
                f"The `order` tuple must have as many entries as there are dimensions in the array. Got {self.order}."
            )
        if len(self.order) != len(set(self.order)):
            raise ValueError(
                f"There must not be duplicates in the `order` tuple. Got {self.order}."
            )
        if not all(0 <= x < ndim for x in self.order):
            raise ValueError(
                f"All entries in the `order` tuple must be between 0 and the number of dimensions in the array. Got {self.order}."
            )
        order = tuple(self.order)

        if order != self.order:
            return replace(self, order=order)
        return self

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            shape=tuple(chunk_spec.shape[self.order[i]] for i in range(chunk_spec.ndim)),
            dtype=chunk_spec.dtype,
            fill_value=chunk_spec.fill_value,
            config=chunk_spec.config,
            prototype=chunk_spec.prototype,
        )

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        inverse_order = np.argsort(self.order)
        return chunk_array.transpose(inverse_order)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        return chunk_array.transpose(self.order)

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        return input_byte_length
