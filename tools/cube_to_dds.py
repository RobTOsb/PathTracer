"""Converts a 3D .cube LUT (Adobe / Resolve format) into a DDS volume texture for the tonemap pass.

The DDS is DXGI_FORMAT_R16G16B16A16_FLOAT, N x N x N, red along x (fastest), green along y, blue along z, as in the
.cube file, with the DX10 header. The renderer lists every .dds in assets/luts in the "Film LUT" menu, named after
the file.

Usage: python tools/cube_to_dds.py input.cube [output.dds]
       (default output: assets/luts/<input name>.dds)
"""

import os
import re
import struct
import sys

DXGI_FORMAT_R16G16B16A16_FLOAT = 10
D3D10_RESOURCE_DIMENSION_TEXTURE3D = 4

DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_DEPTH = 0x800000
DDPF_FOURCC = 0x4
DDSCAPS_COMPLEX = 0x8
DDSCAPS_TEXTURE = 0x1000
DDSCAPS2_VOLUME = 0x200000


def parse_cube(path):
    size = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    values = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key = line.split()[0].upper()
            if key == "TITLE":
                continue
            if key == "LUT_3D_SIZE":
                size = int(line.split()[1])
            elif key == "LUT_1D_SIZE":
                raise ValueError("1D LUTs are not supported, only LUT_3D_SIZE")
            elif key == "DOMAIN_MIN":
                domain_min = [float(v) for v in line.split()[1:4]]
            elif key == "DOMAIN_MAX":
                domain_max = [float(v) for v in line.split()[1:4]]
            elif re.match(r"^[-+0-9.]", line):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"expected 3 values per entry, got: {line!r}")
                values.append(tuple(float(v) for v in parts))
            else:
                print(f"warning: ignoring unknown keyword {key}", file=sys.stderr)
    if size is None:
        raise ValueError("no LUT_3D_SIZE in the file")
    if len(values) != size ** 3:
        raise ValueError(f"expected {size ** 3} entries for size {size}, got {len(values)}")
    if domain_min != [0.0, 0.0, 0.0] or domain_max != [1.0, 1.0, 1.0]:
        raise ValueError(f"only the 0..1 domain is supported (DOMAIN_MIN {domain_min}, DOMAIN_MAX {domain_max})")
    return size, values


def write_dds(path, size, values):
    pitch = size * 8  # 4 halves per texel
    header = struct.pack(
        "<4s7I44x",
        b"DDS ",
        124,
        DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT | DDSD_DEPTH,
        size,  # height
        size,  # width
        pitch,
        size,  # depth
        1,  # mip count
    )
    pixel_format = struct.pack("<2I4s5I", 32, DDPF_FOURCC, b"DX10", 0, 0, 0, 0, 0)
    caps = struct.pack("<4I4x", DDSCAPS_COMPLEX | DDSCAPS_TEXTURE, DDSCAPS2_VOLUME, 0, 0)
    dx10 = struct.pack("<5I", DXGI_FORMAT_R16G16B16A16_FLOAT, D3D10_RESOURCE_DIMENSION_TEXTURE3D, 0, 1, 0)
    assert len(header) + len(pixel_format) + len(caps) == 4 + 124
    texels = bytearray()
    for r, g, b in values:
        texels += struct.pack("<4e", r, g, b, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header + pixel_format + caps + dx10 + texels)


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    source = sys.argv[1]
    if len(sys.argv) == 3:
        target = sys.argv[2]
    else:
        name = os.path.splitext(os.path.basename(source))[0]
        name = re.sub(r"\(\d+\)$", "", name).strip()  # "file(1).cube" from a browser download
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target = os.path.join(root, "assets", "luts", name + ".dds")
    size, values = parse_cube(source)
    write_dds(target, size, values)
    print(f"{source} -> {target} ({size}^3)")


if __name__ == "__main__":
    main()
