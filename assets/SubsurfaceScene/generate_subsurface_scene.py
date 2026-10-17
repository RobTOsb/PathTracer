# Subsurface scattering test scene: a marble cube and a wax cube in a dark open-fronted studio, with
# mean free paths of a third of the cube or more so the scattering is exaggerated. The marble is lit
# from behind by a panel hidden against its back face, so light has to cross it; the wax has a small
# flame hovering over it, so it glows from the top down. A ceiling panel lights the front, and the
# ceiling keeps the sun out. Usage: python generate_subsurface_scene.py <output dir>
import json, math, struct, zlib, os, sys

out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

def png(path, w, h, fn):
    raw = b''.join(b'\x00' + b''.join(bytes(fn(x, y)) for x in range(w)) for y in range(h))
    def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))

# Marble: white with grey-green veins, sin of a diagonal ramp bent by tileable value-noise turbulence.
N = 256
def lattice(seed, cells):
    rnd = [((i * 73856093) ^ (seed * 19349663)) % 1000003 / 1000003.0 for i in range(cells * cells)]
    return lambda i, j: rnd[(j % cells) * cells + (i % cells)]
octaves = [(lattice(o + 1, 4 << o), 4 << o, 0.5 ** o) for o in range(5)]
def turbulence(u, v):
    total = 0.0
    for grid, cells, amp in octaves:
        x, y = u * cells, v * cells
        i, j = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - i, y - j
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        a = grid(i, j) + (grid(i + 1, j) - grid(i, j)) * fx
        b = grid(i, j + 1) + (grid(i + 1, j + 1) - grid(i, j + 1)) * fx
        total += amp * abs(2.0 * (a + (b - a) * fy) - 1.0)
    return total
def marble(x, y):
    u, v = x / N, y / N
    # Whole periods across u and v, so it tiles.
    vein = abs(math.sin(2.0 * math.pi * (2.0 * u + v) + 3.0 * turbulence(u, v)))
    vein = vein ** 0.2  # thin dark lines where the sine crosses zero
    light, dark = (246, 243, 236), (112, 122, 118)
    return tuple(int(dark[c] + (light[c] - dark[c]) * vein) for c in range(3)) + (255,)
png(os.path.join(out, 'marble.png'), N, N, marble)

bufdata = bytearray(); views = []; accessors = []
def add(data, fmt, count, comptype, typ, target, minmax=None):
    while len(bufdata) % 4: bufdata.append(0)
    off = len(bufdata); b = struct.pack('<' + fmt * len(data), *data); bufdata.extend(b)
    views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(b), 'target': target})
    a = {'bufferView': len(views) - 1, 'componentType': comptype, 'count': count, 'type': typ}
    if minmax: a['min'], a['max'] = minmax
    accessors.append(a); return len(accessors) - 1
def mesh_prims(pos, nrm, uv, idx):
    xs = [pos[i::3] for i in range(3)]
    p = add(pos, 'f', len(pos) // 3, 5126, 'VEC3', 34962, ([min(c) for c in xs], [max(c) for c in xs]))
    n = add(nrm, 'f', len(nrm) // 3, 5126, 'VEC3', 34962)
    t = add(uv, 'f', len(uv) // 2, 5126, 'VEC2', 34962)
    i = add(idx, 'I', len(idx), 5125, 'SCALAR', 34963)
    return {'POSITION': p, 'NORMAL': n, 'TEXCOORD_0': t}, i
def quad(sx, sz, uvs=1.0):
    # Facing +Y, counter-clockwise seen from above.
    pos = [-sx, 0, -sz,  sx, 0, -sz,  sx, 0, sz,  -sx, 0, sz]; nrm = [0, 1, 0] * 4
    uv = [0, 0, uvs, 0, uvs, uvs, 0, uvs]; idx = [0, 2, 1, 0, 3, 2]
    return pos, nrm, uv, idx
def box(hx, hy, hz):
    # Counter-clockwise seen from outside, one uv square per face.
    h = (hx, hy, hz); pos = []; nrm = []; uv = []; idx = []
    for axis in range(3):
        for sgn in (-1, 1):
            n = [0, 0, 0]; n[axis] = sgn; u = [0, 0, 0]; v = [0, 0, 0]
            u[(axis + 1) % 3] = 1; v[(axis + 2) % 3] = 1
            if sgn < 0: u, v = v, u
            base = len(pos) // 3
            for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                pos += [h[k] * (n[k] + a * u[k] + b * v[k]) for k in range(3)]; nrm += n; uv += [(a + 1) / 2, (b + 1) / 2]
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return pos, nrm, uv, idx

# Column-major 4x4 transforms, composed on the CPU since the loader takes each node's own transform.
def mul(a, b): return [sum(a[k * 4 + r] * b[c * 4 + k] for k in range(4)) for c in range(4) for r in range(4)]
def translate(x, y, z): return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]
def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]
def rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]
def chain(*ms):
    m = ms[0]
    for n in ms[1:]: m = mul(m, n)
    return m

meshes = []
def add_mesh(name, geo, material):
    attrs, ind = mesh_prims(*geo)
    meshes.append({'name': name, 'primitives': [{'attributes': attrs, 'indices': ind, 'material': material}]})
    return len(meshes) - 1

def emissive(name, colour, strength):
    return {'name': name, 'pbrMetallicRoughness': {'baseColorFactor': [0, 0, 0, 1], 'metallicFactor': 0.0, 'roughnessFactor': 1.0},
            'emissiveFactor': colour, 'extensions': {'KHR_materials_emissive_strength': {'emissiveStrength': strength}}}

materials = [
    {'name': 'Studio', 'pbrMetallicRoughness': {'baseColorFactor': [0.2, 0.2, 0.21, 1], 'metallicFactor': 0.0, 'roughnessFactor': 0.9}},
    # Polished marble: nearly white, faintly warm deep down. As in Blender, the base colour (here its
    # texture) is the medium's multiple-scattering albedo, and radius * scale its mean free path.
    {'name': 'Marble', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicFactor': 0.0, 'roughnessFactor': 0.12},
     'extensions': {'EXT_materials_subsurface': {'subsurfaceFactor': 1.0, 'subsurfaceRadius': [1.0, 0.8, 0.6], 'subsurfaceScale': 0.25,
                                                 'subsurfaceIor': 1.4, 'subsurfaceAnisotropy': 0.0, 'subsurfaceMethod': 'RANDOM_WALK'}}},
    # Candle wax: red light travels far further than blue, and scatters a little forwards.
    {'name': 'Wax', 'pbrMetallicRoughness': {'baseColorFactor': [0.98, 0.72, 0.38, 1], 'metallicFactor': 0.0, 'roughnessFactor': 0.35},
     'extensions': {'EXT_materials_subsurface': {'subsurfaceFactor': 1.0, 'subsurfaceRadius': [1.0, 0.45, 0.2], 'subsurfaceScale': 0.35,
                                                 'subsurfaceIor': 1.4, 'subsurfaceAnisotropy': 0.2, 'subsurfaceMethod': 'RANDOM_WALK'}}},
    emissive('KeyLight', [1.0, 0.97, 0.92], 5.0),
    emissive('BackLight', [1.0, 0.98, 0.95], 30.0),
    emissive('Flame', [1.0, 0.55, 0.18], 80.0),
]

H = 0.4  # half the cube's edge
studio = (2.2, 2.4, 1.6)  # half width, height, half depth
m_floor = add_mesh('Floor', quad(studio[0], studio[2], 4), 0)
m_wall_back = add_mesh('WallBack', quad(studio[0], studio[1] / 2, 4), 0)
m_wall_side = add_mesh('WallSide', quad(studio[1] / 2, studio[2], 4), 0)
m_marble = add_mesh('MarbleCube', box(H, H, H), 1)
m_wax = add_mesh('WaxCube', box(H, H, H), 2)
m_key = add_mesh('KeyLight', quad(0.7, 0.3), 3)
m_back = add_mesh('BackLight', quad(0.25, 0.25), 4)
m_flame = add_mesh('Flame', box(0.025, 0.05, 0.025), 5)

marble_at = chain(translate(-0.85, H, 0.0), rot_y(35))
wax_at = chain(translate(0.85, H, 0.1), rot_y(-25))
nodes = [
    {'name': 'Floor', 'mesh': m_floor},
    {'name': 'Ceiling', 'mesh': m_floor, 'matrix': chain(translate(0, studio[1], 0), rot_x(180))},
    {'name': 'WallBack', 'mesh': m_wall_back, 'matrix': chain(translate(0, studio[1] / 2, -studio[2]), rot_x(90))},
    # Quarter turns about Z: +Y (the quad's front) to +X on the left wall, to -X on the right.
    {'name': 'WallLeft', 'mesh': m_wall_side, 'matrix': chain(translate(-studio[0], studio[1] / 2, 0), [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])},
    {'name': 'WallRight', 'mesh': m_wall_side, 'matrix': chain(translate(studio[0], studio[1] / 2, 0), [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])},
    {'name': 'MarbleCube', 'mesh': m_marble, 'matrix': marble_at},
    {'name': 'WaxCube', 'mesh': m_wax, 'matrix': wax_at},
    # Facing down from the ceiling, in front of the cubes.
    {'name': 'KeyLight', 'mesh': m_key, 'matrix': chain(translate(0, studio[1] - 0.01, 0.9), rot_x(180))},
    # 2 cm behind the marble's back face, facing it, smaller than the face so the cube hides it.
    {'name': 'BackLight', 'mesh': m_back, 'matrix': chain(marble_at, translate(0, 0, -H - 0.02), rot_x(90))},
    # Hovering over the middle of the wax's top face.
    {'name': 'Flame', 'mesh': m_flame, 'matrix': chain(wax_at, translate(0, H + 0.09, 0))},
]

gltf = {'asset': {'version': '2.0', 'generator': 'generate_subsurface_scene.py'},
        'extensionsUsed': ['EXT_materials_subsurface', 'KHR_materials_emissive_strength'],
        'scene': 0, 'scenes': [{'nodes': list(range(len(nodes)))}], 'nodes': nodes, 'meshes': meshes, 'materials': materials,
        'textures': [{'source': 0}], 'images': [{'uri': 'marble.png'}],
        'buffers': [{'uri': 'scene.bin', 'byteLength': len(bufdata)}], 'bufferViews': views, 'accessors': accessors}
open(os.path.join(out, 'scene.bin'), 'wb').write(bufdata)
json.dump(gltf, open(os.path.join(out, 'scene.gltf'), 'w'), indent=1)
print('ok', len(bufdata))
