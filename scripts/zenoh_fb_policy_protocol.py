#!/usr/bin/env python3
# ruff: noqa
# Verbatim copy of derroboter's src/derroboter/py/nodes/zenoh_fb_policy_protocol.py (last changed in derroboter
# 6d3a45382, 2026-05-18), so derroboter_policy_server.py needs no derroboter checkout. Payloads carry
# SCHEMA_VERSION and decode_payload rejects any other version: a protocol change fails loudly, then re-copy it.
"""FlatBuffer v1 payload for Zenoh SHM policy requests/responses.

The public helpers intentionally keep the old Python boundary:

  encode_payload(obs_or_action) -> bytes
  decode_payload(payload) -> obs_or_action

Internally, payloads are encoded as ``Dbot.Policy.PolicyEnvelope`` from
``policy_payload.fbs``. The envelope carries schema version, message kind,
optional request/response metadata, tensor metadata, scalar JSON fields, and one
contiguous raw byte blob for tensor data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import flatbuffers
import numpy as np
import torch


SCHEMA_VERSION = 1

MESSAGE_KIND_GENERIC = 0
MESSAGE_KIND_REQUEST = 1
MESSAGE_KIND_RESPONSE = 2
MESSAGE_KIND_ERROR = 3

DTYPE_UNKNOWN = 0
DTYPE_BOOL = 1
DTYPE_UINT8 = 2
DTYPE_INT8 = 3
DTYPE_UINT16 = 4
DTYPE_INT16 = 5
DTYPE_UINT32 = 6
DTYPE_INT32 = 7
DTYPE_UINT64 = 8
DTYPE_INT64 = 9
DTYPE_FLOAT16 = 10
DTYPE_FLOAT32 = 11
DTYPE_FLOAT64 = 12

LAYOUT_ROW_MAJOR = 0
LAYOUT_HWC = 1
LAYOUT_CHW = 2

ARRAY_KIND_UNKNOWN = 0
ARRAY_KIND_NUMPY = 1
ARRAY_KIND_TORCH = 2
ARRAY_KIND_CPP = 3

RESPONSE_STATUS_OK = 0
RESPONSE_STATUS_ERROR = 1

_DTYPE_TO_ENUM = {
    "bool": DTYPE_BOOL,
    "uint8": DTYPE_UINT8,
    "int8": DTYPE_INT8,
    "uint16": DTYPE_UINT16,
    "int16": DTYPE_INT16,
    "uint32": DTYPE_UINT32,
    "int32": DTYPE_INT32,
    "uint64": DTYPE_UINT64,
    "int64": DTYPE_INT64,
    "float16": DTYPE_FLOAT16,
    "float32": DTYPE_FLOAT32,
    "float64": DTYPE_FLOAT64,
}

_ENUM_TO_DTYPE = {value: key for key, value in _DTYPE_TO_ENUM.items()}


@dataclass
class _TensorSpec:
    path: list[str]
    dtype: int
    dtype_name: str
    shape: list[int]
    offset: int
    nbytes: int
    array_kind: int
    layout: int = LAYOUT_ROW_MAJOR
    byte_order: str = "little"


@dataclass
class _ScalarSpec:
    path: list[str]
    json_value: str


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    return value


def _array_to_numpy(value: Any) -> tuple[np.ndarray, int]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().numpy(), ARRAY_KIND_TORCH
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value), ARRAY_KIND_NUMPY
    raise TypeError(type(value).__name__)


def _infer_layout(path: list[str], shape: list[int]) -> int:
    path_text = ".".join(path).lower()
    if "image" in path_text and len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            return LAYOUT_HWC
        if shape[0] in (1, 3, 4):
            return LAYOUT_CHW
    return LAYOUT_ROW_MAJOR


def _walk(
    value: Any,
    path: list[str],
    tensors: list[_TensorSpec],
    scalars: list[_ScalarSpec],
    blobs: list[bytes],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _walk(child, [*path, str(key)], tensors, scalars, blobs)
        return

    if isinstance(value, (torch.Tensor, np.ndarray)):
        array, array_kind = _array_to_numpy(value)
        dtype_name = str(array.dtype)
        shape = list(array.shape)
        offset = sum(len(blob) for blob in blobs)
        data = array.tobytes(order="C")
        blobs.append(data)
        tensors.append(
            _TensorSpec(
                path=path,
                dtype=_DTYPE_TO_ENUM.get(dtype_name, DTYPE_UNKNOWN),
                dtype_name=dtype_name,
                shape=shape,
                offset=offset,
                nbytes=len(data),
                array_kind=array_kind,
                layout=_infer_layout(path, shape),
            )
        )
        return

    scalars.append(
        _ScalarSpec(
            path=path,
            json_value=json.dumps(_json_default(value), separators=(",", ":")),
        )
    )


def _infer_message_kind(value: Any) -> int:
    if isinstance(value, Mapping):
        return MESSAGE_KIND_REQUEST
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return MESSAGE_KIND_RESPONSE
    return MESSAGE_KIND_GENERIC


def _normalize_message_kind(message_kind: int | str | None, value: Any) -> int:
    if message_kind is None:
        return _infer_message_kind(value)
    if isinstance(message_kind, int):
        return message_kind
    mapping = {
        "generic": MESSAGE_KIND_GENERIC,
        "request": MESSAGE_KIND_REQUEST,
        "response": MESSAGE_KIND_RESPONSE,
        "error": MESSAGE_KIND_ERROR,
    }
    try:
        return mapping[message_kind.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported policy message kind: {message_kind!r}") from exc


def _start_shape_vector(builder: flatbuffers.Builder, shape: list[int]) -> int:
    builder.StartVector(8, len(shape), 8)
    for dim in reversed(shape):
        builder.PrependUint64(int(dim))
    return builder.EndVector()


def _build_offset_vector(builder: flatbuffers.Builder, offsets: list[int]) -> int:
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _build_tensor(builder: flatbuffers.Builder, spec: _TensorSpec) -> int:
    path = builder.CreateString(json.dumps(spec.path, separators=(",", ":")))
    dtype_name = builder.CreateString(spec.dtype_name)
    shape = _start_shape_vector(builder, spec.shape)
    byte_order = builder.CreateString(spec.byte_order)

    builder.StartObject(9)
    builder.PrependUOffsetTRelativeSlot(0, path, 0)
    builder.PrependInt8Slot(1, spec.dtype, 0)
    builder.PrependUOffsetTRelativeSlot(2, dtype_name, 0)
    builder.PrependUOffsetTRelativeSlot(3, shape, 0)
    builder.PrependUint64Slot(4, spec.offset, 0)
    builder.PrependUint64Slot(5, spec.nbytes, 0)
    builder.PrependInt8Slot(6, spec.array_kind, 0)
    builder.PrependInt8Slot(7, spec.layout, 0)
    builder.PrependUOffsetTRelativeSlot(8, byte_order, 0)
    return builder.EndObject()


def _build_scalar(builder: flatbuffers.Builder, spec: _ScalarSpec) -> int:
    path = builder.CreateString(json.dumps(spec.path, separators=(",", ":")))
    json_value = builder.CreateString(spec.json_value)
    builder.StartObject(2)
    builder.PrependUOffsetTRelativeSlot(0, path, 0)
    builder.PrependUOffsetTRelativeSlot(1, json_value, 0)
    return builder.EndObject()


def _build_request(
    builder: flatbuffers.Builder,
    request_id: str | None,
    robot_id: str | None,
    policy_key: str | None,
) -> int:
    request_id_offset = builder.CreateString(request_id or "")
    robot_id_offset = builder.CreateString(robot_id or "")
    policy_key_offset = builder.CreateString(policy_key or "")
    builder.StartObject(3)
    builder.PrependUOffsetTRelativeSlot(0, request_id_offset, 0)
    builder.PrependUOffsetTRelativeSlot(1, robot_id_offset, 0)
    builder.PrependUOffsetTRelativeSlot(2, policy_key_offset, 0)
    return builder.EndObject()


def _build_response(
    builder: flatbuffers.Builder,
    request_id: str | None,
    status: int,
    error_message: str | None,
) -> int:
    request_id_offset = builder.CreateString(request_id or "")
    error_offset = builder.CreateString(error_message or "")
    builder.StartObject(3)
    builder.PrependUOffsetTRelativeSlot(0, request_id_offset, 0)
    builder.PrependInt8Slot(1, status, RESPONSE_STATUS_OK)
    builder.PrependUOffsetTRelativeSlot(2, error_offset, 0)
    return builder.EndObject()


def encode_payload(
    value: Any,
    *,
    message_kind: int | str | None = None,
    request_id: str | None = None,
    robot_id: str | None = None,
    policy_key: str | None = None,
    response_status: int = RESPONSE_STATUS_OK,
    error_message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    tensors: list[_TensorSpec] = []
    scalars: list[_ScalarSpec] = []
    blobs: list[bytes] = []
    _walk(value, [], tensors, scalars, blobs)
    blob = b"".join(blobs)
    kind = _normalize_message_kind(message_kind, value)

    builder = flatbuffers.Builder(max(1024, len(blob) + 4096))
    blob_vec = builder.CreateByteVector(blob)
    tensor_offsets = [_build_tensor(builder, spec) for spec in tensors]
    scalar_offsets = [_build_scalar(builder, spec) for spec in scalars]
    tensors_vec = _build_offset_vector(builder, tensor_offsets)
    scalars_vec = _build_offset_vector(builder, scalar_offsets)
    request_offset = _build_request(builder, request_id, robot_id, policy_key)
    response_offset = _build_response(builder, request_id, response_status, error_message)
    metadata_json = builder.CreateString(
        json.dumps(_json_default(metadata or {}), separators=(",", ":"))
    )

    builder.StartObject(8)
    builder.PrependUint32Slot(0, SCHEMA_VERSION, 0)
    builder.PrependInt8Slot(1, kind, MESSAGE_KIND_GENERIC)
    builder.PrependUOffsetTRelativeSlot(2, request_offset, 0)
    builder.PrependUOffsetTRelativeSlot(3, response_offset, 0)
    builder.PrependUOffsetTRelativeSlot(4, tensors_vec, 0)
    builder.PrependUOffsetTRelativeSlot(5, scalars_vec, 0)
    builder.PrependUOffsetTRelativeSlot(6, blob_vec, 0)
    builder.PrependUOffsetTRelativeSlot(7, metadata_json, 0)
    root = builder.EndObject()
    builder.Finish(root)
    return bytes(builder.Output())


def encode_request(value: Any, **kwargs: Any) -> bytes:
    return encode_payload(value, message_kind=MESSAGE_KIND_REQUEST, **kwargs)


def encode_response(value: Any, **kwargs: Any) -> bytes:
    return encode_payload(value, message_kind=MESSAGE_KIND_RESPONSE, **kwargs)


class _Table:
    def __init__(self, buf: bytes | bytearray | memoryview, pos: int) -> None:
        import flatbuffers.table

        self._tab = flatbuffers.table.Table(buf, pos)

    def _offset(self, slot: int) -> int:
        return self._tab.Offset(4 + slot * 2)

    def string(self, slot: int) -> str:
        o = self._offset(slot)
        if o == 0:
            return ""
        return self._tab.String(o + self._tab.Pos).decode("utf-8")

    def uint32(self, slot: int) -> int:
        o = self._offset(slot)
        if o == 0:
            return 0
        return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)

    def uint64(self, slot: int) -> int:
        o = self._offset(slot)
        if o == 0:
            return 0
        return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)

    def int8(self, slot: int) -> int:
        o = self._offset(slot)
        if o == 0:
            return 0
        return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)

    def vector_len(self, slot: int) -> int:
        o = self._offset(slot)
        if o == 0:
            return 0
        return self._tab.VectorLen(o)

    def vector_table(self, slot: int, index: int) -> "_Table":
        o = self._offset(slot)
        start = self._tab.Vector(o)
        pos = self._tab.Indirect(start + index * 4)
        return _Table(self._tab.Bytes, pos)

    def vector_uint64(self, slot: int, index: int) -> int:
        o = self._offset(slot)
        start = self._tab.Vector(o)
        return self._tab.Get(flatbuffers.number_types.Uint64Flags, start + index * 8)

    def byte_vector(self, slot: int) -> memoryview:
        o = self._offset(slot)
        if o == 0:
            return memoryview(b"")
        start = self._tab.Vector(o)
        length = self._tab.VectorLen(o)
        return memoryview(self._tab.Bytes)[start : start + length]


def _root(buf: bytes | bytearray | memoryview) -> _Table:
    import flatbuffers.encode
    import flatbuffers.packer

    n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, 0)
    return _Table(buf, n)


def _assign_path(root: dict[str, Any], parts: list[str], value: Any) -> None:
    current = root
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _dtype_name(dtype: int, name: str) -> str:
    if name:
        return name
    return _ENUM_TO_DTYPE.get(dtype, "uint8")


def decode_payload(payload: bytes | bytearray | memoryview, *, restore_torch: bool = True) -> Any:
    envelope = _root(payload)
    version = envelope.uint32(0)
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported zenoh_fb_shm schema version: {version}")

    result: dict[str, Any] = {}
    root_value: Any = None
    has_root_value = False
    blob = envelope.byte_vector(6)

    for i in range(envelope.vector_len(5)):
        scalar = envelope.vector_table(5, i)
        parts = json.loads(scalar.string(0))
        value = json.loads(scalar.string(1))
        if not parts:
            root_value = value
            has_root_value = True
        else:
            _assign_path(result, parts, value)

    for i in range(envelope.vector_len(4)):
        tensor = envelope.vector_table(4, i)
        parts = json.loads(tensor.string(0))
        dtype = np.dtype(_dtype_name(int(tensor.int8(1)), tensor.string(2)))
        shape = [int(tensor.vector_uint64(3, j)) for j in range(tensor.vector_len(3))]
        offset = int(tensor.uint64(4))
        nbytes = int(tensor.uint64(5))
        array_kind = int(tensor.int8(6))
        if offset + nbytes > len(blob):
            raise ValueError(
                f"Tensor {parts!r} byte range {offset}:{offset + nbytes} exceeds blob size {len(blob)}"
            )
        raw = blob[offset : offset + nbytes]
        array = np.frombuffer(raw, dtype=dtype).reshape(shape)
        array = np.array(array, copy=True)
        if restore_torch and array_kind == ARRAY_KIND_TORCH:
            value = torch.from_numpy(array)
        else:
            value = array
        if not parts:
            root_value = value
            has_root_value = True
        else:
            _assign_path(result, parts, value)

    return root_value if has_root_value else result
