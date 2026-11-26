from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Every PLY scalar type, with both the old names ("float") and the newer sized names
# ("float32"). Values are NumPy type codes without the byte-order prefix.
PLY_TO_NUMPY = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}
# For writing new files we use the old names, which is what 3DGS tools write.
NUMPY_TO_PLY = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
                "i4": "int", "u4": "uint", "f4": "float", "f8": "double"}

FORMAT_TO_BYTE_ORDER = {"binary_little_endian": "<", "binary_big_endian": ">", "ascii": "<"}


@dataclass
class PlyProperty:
    name: str
    type_name: str  # exactly as written in the header, e.g. "float" or "float32"

    @property
    def numpy_code(self) -> str:
        return PLY_TO_NUMPY[self.type_name]


@dataclass
class PlyElement:
    name: str
    count: int
    properties: list[PlyProperty] = field(default_factory=list)


@dataclass
class PlyHeader:
    format: str                 # "binary_little_endian" | "binary_big_endian" | "ascii"
    version: str                # almost always "1.0"
    comments: list[str]
    obj_info: list[str]
    elements: list[PlyElement]
    raw_bytes: bytes            # the header exactly as read, up to and including end_header\n

    def vertex_dtype(self) -> np.dtype:
        """Structured dtype of one vertex row, in file order and file byte order."""
        order = FORMAT_TO_BYTE_ORDER[self.format]
        vertex = self.element("vertex")
        return np.dtype([(p.name, order + p.numpy_code) for p in vertex.properties])

    def element(self, name: str) -> PlyElement:
        for element in self.elements:
            if element.name == name:
                return element
        raise KeyError(f"PLY file has no '{name}' element")


def read_header(f) -> PlyHeader:
    """Parse the header from an open binary file, leaving the file positioned at the body."""
    raw_lines = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Reached end of file before 'end_header'")
        raw_lines.append(line)
        if line.strip() == b"end_header":
            break

    if raw_lines[0].strip() != b"ply":
        raise ValueError("Not a PLY file (first line is not 'ply')")

    fmt, version = None, None
    comments, obj_info, elements = [], [], []
    for raw in raw_lines[1:-1]:
        text = raw.decode("ascii").rstrip("\r\n")
        words = text.split()
        if not words:
            continue
        keyword = words[0]
        if keyword == "format":
            fmt, version = words[1], words[2]
            if fmt not in FORMAT_TO_BYTE_ORDER:
                raise ValueError(f"Unknown PLY format '{fmt}'")
        elif keyword == "comment":
            comments.append(text[len("comment "):])
        elif keyword == "obj_info":
            obj_info.append(text[len("obj_info "):])
        elif keyword == "element":
            elements.append(PlyElement(name=words[1], count=int(words[2])))
        elif keyword == "property":
            if not elements:
                raise ValueError("'property' line before any 'element' line")
            if words[1] == "list":
                # List properties (e.g. mesh faces) have a variable size per row, so they
                # cannot be a structured dtype. Splats never use them for vertices.
                raise NotImplementedError("List properties are not supported yet (M1)")
            if words[1] not in PLY_TO_NUMPY:
                raise ValueError(f"Unknown PLY property type '{words[1]}'")
            elements[-1].properties.append(PlyProperty(name=words[2], type_name=words[1]))
        else:
            raise ValueError(f"Unknown PLY header line: {text!r}")

    if fmt is None:
        raise ValueError("PLY header has no 'format' line")
    return PlyHeader(fmt, version, comments, obj_info, elements, b"".join(raw_lines))


def read_ply(path: str | Path) -> tuple[PlyHeader, np.ndarray]:
    """Read a splat PLY. Returns the header and the vertex rows as a structured array.

    The array keeps the file's exact property names, order and dtypes (including byte
    order), so nothing is converted or rounded.
    """
    with open(path, "rb") as f:
        header = read_header(f)
        other = [e.name for e in header.elements if e.name != "vertex"]
        if other:
            raise NotImplementedError(f"Elements other than 'vertex' not supported yet (M1): {other}")

        dtype = header.vertex_dtype()
        count = header.element("vertex").count
        if header.format == "ascii":
            data = np.loadtxt(f, dtype=dtype, max_rows=count, ndmin=1)
        else:
            data = np.fromfile(f, dtype=dtype, count=count)

        if len(data) != count:
            raise ValueError(f"Header says {count} vertices but the body holds {len(data)}")
    return header, data


def write_ply(
    path: str | Path,
    data: np.ndarray,
    comments: list[str] = (),
    overwrite: bool = False,
) -> None:
    """Write a structured array as a new binary little-endian PLY file.

    The header is built from the array's dtype, so property names and order follow the
    array's fields. `overwrite=False` by default, so a stage cannot accidentally replace
    an existing file such as the input splat (SPEC 3.2 rule 3).
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {c}" for c in comments]
    lines.append(f"element vertex {len(data)}")
    for name in data.dtype.names:
        code = data.dtype[name].str[1:]  # e.g. "<f4" -> "f4"
        lines.append(f"property {NUMPY_TO_PLY[code]} {name}")
    lines.append("end_header")

    # Convert to little-endian only if needed (a no-op on the usual little-endian data).
    little = data.astype(data.dtype.newbyteorder("<"), copy=False)
    with open(path, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))
        little.tofile(f)


def sh_degree_from_properties(names: list[str] | tuple[str, ...]) -> int:
    """Spherical-harmonics degree from the number of `f_rest_*` properties.

    Degree d has (d+1)^2 coefficients per colour channel. The first (the DC term) is
    stored as f_dc_0..2, the other (d+1)^2 - 1 per channel as f_rest_*, so there are
    3 * ((d+1)^2 - 1) f_rest properties: 0, 9, 24 or 45 for degrees 0 to 3.
    """
    n_rest = sum(1 for n in names if n.startswith("f_rest_"))
    for degree in range(5):
        if n_rest == 3 * ((degree + 1) ** 2 - 1):
            return degree
    raise ValueError(f"{n_rest} f_rest_* properties does not match any SH degree")
