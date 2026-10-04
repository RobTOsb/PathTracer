import json, math, struct, zlib, os, sys
out = sys.argv[1]
def png(path, w, h, fn):
    raw = b''.join(b'\x00' + b''.join(bytes(fn(x, y)) for x in range(w)) for y in range(h))
    def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
# checker albedo (sRGB)
png(os.path.join(out, 'checker.png'), 256, 256, lambda x, y: (230,230,230,255) if ((x//32)+(y//32))%2==0 else (60,60,70,255))
# roughness/metal: G = roughness stripes, B = 0
png(os.path.join(out, 'roughmetal.png'), 64, 64, lambda x, y: (0, 60 if (y//16)%2==0 else 230, 0, 255))
# normal map: hemispherical bumps per cell
def nm(x, y):
    cx = (x % 32) - 15.5; cy = (y % 32) - 15.5; r = 12.0
    d2 = cx*cx + cy*cy
    if d2 < r*r:
        z = math.sqrt(r*r - d2); n = (cx/r, -cy/r, z/r)   # +Y up in tangent space (V down in image)
    else: n = (0.0, 0.0, 1.0)
    return tuple(int(round((c*0.5+0.5)*255)) for c in n) + (255,)
png(os.path.join(out, 'normal.png'), 256, 256, nm)

bufdata = bytearray(); views = []; accessors = []
def add(data, fmt, count, comptype, typ, target, minmax=None):
    while len(bufdata) % 4: bufdata.append(0)
    off = len(bufdata); b = struct.pack('<' + fmt * (len(data)), *data); bufdata.extend(b)
    views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(b), 'target': target})
    a = {'bufferView': len(views)-1, 'componentType': comptype, 'count': count, 'type': typ}
    if minmax: a['min'], a['max'] = minmax
    accessors.append(a); return len(accessors)-1
def mesh_prims(pos, nrm, uv, idx):
    xs = [pos[i::3] for i in range(3)]
    p = add(pos, 'f', len(pos)//3, 5126, 'VEC3', 34962, ([min(c) for c in xs], [max(c) for c in xs]))
    n = add(nrm, 'f', len(nrm)//3, 5126, 'VEC3', 34962)
    t = add(uv, 'f', len(uv)//2, 5126, 'VEC2', 34962)
    i = add(idx, 'I', len(idx), 5125, 'SCALAR', 34963)
    return {'POSITION': p, 'NORMAL': n, 'TEXCOORD_0': t}, i
def plane(s, uvs):
    pos = [-s,0,-s,  s,0,-s,  s,0,s,  -s,0,s]; nrm = [0,1,0]*4
    uv = [0,0, uvs,0, uvs,uvs, 0,uvs]; idx = [0,2,1, 0,3,2]   # CCW seen from +Y
    return pos, nrm, uv, idx
def sphere(r, seg=48, rings=24):
    pos=[]; nrm=[]; uv=[]; idx=[]
    for j in range(rings+1):
        v = j/rings; th = v*math.pi
        for i in range(seg+1):
            u = i/seg; ph = u*2*math.pi
            n = (math.sin(th)*math.cos(ph), math.cos(th), -math.sin(th)*math.sin(ph))
            pos += [c*r for c in n]; nrm += n; uv += [u*4, v*2]
    for j in range(rings):
        for i in range(seg):
            a = j*(seg+1)+i; b = a+seg+1
            idx += [a, b, a+1,  a+1, b, b+1]
    return pos, nrm, uv, idx
def box(h):
    pos=[]; nrm=[]; uv=[]; idx=[]
    for axis in range(3):
        for sgn in (-1, 1):
            n=[0,0,0]; n[axis]=sgn; u=[0,0,0]; v=[0,0,0]
            u[(axis+1)%3]=1; v[(axis+2)%3]=1
            if sgn<0: u,v=v,u
            base=len(pos)//3
            for (a,b) in ((-1,-1),(1,-1),(1,1),(-1,1)):
                pos += [h*(n[k]+a*u[k]+b*v[k]) for k in range(3)]; nrm += n; uv += [(a+1)/2,(b+1)/2]
            idx += [base,base+1,base+2, base,base+2,base+3]
    return pos, nrm, uv, idx

meshes = []
def add_mesh(name, geo, mat):
    attrs, ind = mesh_prims(*geo)
    meshes.append({'name': name, 'primitives': [{'attributes': attrs, 'indices': ind, 'material': mat}]})
    return len(meshes)-1
m_plane = add_mesh('Ground', plane(10, 10), 0)
m_sphere_r = add_mesh('SphereRed', sphere(1.0), 1)
m_sphere_g = add_mesh('SphereGold', sphere(1.0), 2)
m_sphere_c = add_mesh('SphereCoated', sphere(1.0), 3)
m_box = add_mesh('Box', box(0.75), 4)

materials = [
    {'name': 'Ground', 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'metallicRoughnessTexture': {'index': 1}, 'metallicFactor': 1.0, 'roughnessFactor': 1.0}, 'normalTexture': {'index': 2, 'scale': 1.0}},
    {'name': 'RedDiffuse', 'pbrMetallicRoughness': {'baseColorFactor': [0.8, 0.1, 0.08, 1], 'metallicFactor': 0.0, 'roughnessFactor': 0.8}},
    {'name': 'Gold', 'pbrMetallicRoughness': {'baseColorFactor': [1.0, 0.78, 0.34, 1], 'metallicFactor': 1.0, 'roughnessFactor': 0.25}},
    {'name': 'BlueGlossy', 'pbrMetallicRoughness': {'baseColorFactor': [0.1, 0.25, 0.8, 1], 'metallicFactor': 0.0, 'roughnessFactor': 0.05}},
    {'name': 'Chrome', 'pbrMetallicRoughness': {'baseColorFactor': [0.95, 0.95, 0.95, 1], 'metallicFactor': 1.0, 'roughnessFactor': 0.02}},
]
q = (0.0, math.sin(math.radians(30)/2), 0.0, math.cos(math.radians(30)/2))  # 30 deg about Y
# column-major matrix: rotate 45 deg about Y, translate (3, 0.75, -1)
c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
mat = [c,0,-s,0,  0,1,0,0,  s,0,c,0,  3,0.75,-1,1]
nodes = [
    {'name': 'Ground', 'mesh': m_plane},
    {'name': 'Red', 'mesh': m_sphere_r, 'translation': [-2.5, 1, 0]},
    {'name': 'Gold', 'mesh': m_sphere_g, 'translation': [0, 1, 0]},
    {'name': 'BlueStretched', 'mesh': m_sphere_c, 'translation': [2.5, 0.6, 2.0], 'rotation': list(q), 'scale': [1.0, 0.6, 0.5]},
    {'name': 'BoxMatrix', 'mesh': m_box, 'matrix': mat},
    {'name': 'BoxMirrored', 'mesh': m_box, 'translation': [-3.0, 0.75, -2.5], 'scale': [-1, 1, 1], 'rotation': list(q)},
]
gltf = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': list(range(len(nodes)))}], 'nodes': nodes, 'meshes': meshes,
        'materials': materials, 'textures': [{'source': 0}, {'source': 1}, {'source': 2}],
        'images': [{'uri': 'checker.png'}, {'uri': 'roughmetal.png'}, {'uri': 'normal.png'}],
        'buffers': [{'uri': 'scene.bin', 'byteLength': len(bufdata)}], 'bufferViews': views, 'accessors': accessors}
open(os.path.join(out, 'scene.bin'), 'wb').write(bufdata)
json.dump(gltf, open(os.path.join(out, 'scene.gltf'), 'w'), indent=1)
print('ok', len(bufdata))
