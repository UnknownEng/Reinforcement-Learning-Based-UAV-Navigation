"""
raycast_jit.py
--------------
Numba-compiled kernels for World3D: ray casting, distance queries and occupancy
rasterization. Same geometry as the NumPy reference code in world3d.py (which is
used automatically when numba is not installed).
"""

import math

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - fallback path
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap if not args or not callable(args[0]) else args[0]


@njit(cache=True, fastmath=True)
def raycast(o, dirs, cyl, sph, box, cap, max_range):
    n_rays = dirs.shape[0]
    out = np.empty(n_rays)
    ox, oy, oz = o[0], o[1], o[2]
    for i in range(n_rays):
        dx, dy, dz = dirs[i, 0], dirs[i, 1], dirs[i, 2]
        best = max_range
        inside = False
        # spheres
        for j in range(sph.shape[0]):
            cx, cy, cz, r = sph[j, 0], sph[j, 1], sph[j, 2], sph[j, 3]
            qx, qy, qz = ox - cx, oy - cy, oz - cz
            c = qx * qx + qy * qy + qz * qz - r * r
            if c <= 0.0:
                inside = True
                break
            b = dx * qx + dy * qy + dz * qz
            if b >= 0.0:
                continue
            disc = b * b - c
            if disc < 0.0:
                continue
            t = -b - math.sqrt(disc)
            if t < best:
                best = t
        # vertical finite cylinders
        if not inside:
            for j in range(cyl.shape[0]):
                cx, cy, r, z0, z1 = cyl[j, 0], cyl[j, 1], cyl[j, 2], cyl[j, 3], cyl[j, 4]
                qx, qy = ox - cx, oy - cy
                cc = qx * qx + qy * qy - r * r
                if cc <= 0.0 and z0 <= oz <= z1:
                    inside = True
                    break
                a = dx * dx + dy * dy
                if a > 1e-12 and cc > 0.0:
                    hb = qx * dx + qy * dy
                    disc = hb * hb - a * cc
                    if disc >= 0.0:
                        t = (-hb - math.sqrt(disc)) / a
                        if 0.0 < t < best:
                            z = oz + t * dz
                            if z0 <= z <= z1:
                                best = t
                if abs(dz) > 1e-12:
                    for k in range(2):
                        zc = z0 if k == 0 else z1
                        t = (zc - oz) / dz
                        if 0.0 < t < best:
                            hx, hy = qx + t * dx, qy + t * dy
                            if hx * hx + hy * hy <= r * r:
                                best = t
        # yawed boxes (slab test in the box frame)
        if not inside:
            for j in range(box.shape[0]):
                cyw, syw = math.cos(box[j, 6]), math.sin(box[j, 6])
                rx, ry, rz = ox - box[j, 0], oy - box[j, 1], oz - box[j, 2]
                lo = (cyw * rx + syw * ry, -syw * rx + cyw * ry, rz)
                ld = (cyw * dx + syw * dy, -syw * dx + cyw * dy, dz)
                h = (box[j, 3], box[j, 4], box[j, 5])
                tmin, tmax = -1e30, 1e30
                miss = False
                for ax in range(3):
                    if abs(ld[ax]) < 1e-12:
                        if lo[ax] < -h[ax] or lo[ax] > h[ax]:
                            miss = True
                            break
                    else:
                        t1 = (-h[ax] - lo[ax]) / ld[ax]
                        t2 = (h[ax] - lo[ax]) / ld[ax]
                        if t1 > t2:
                            t1, t2 = t2, t1
                        if t1 > tmin:
                            tmin = t1
                        if t2 < tmax:
                            tmax = t2
                if miss or tmax < max(tmin, 0.0):
                    continue
                if tmin <= 0.0:
                    inside = True
                    break
                if tmin < best:
                    best = tmin
        # capsules
        if not inside:
            for j in range(cap.shape[0]):
                ax_, ay_, az_ = cap[j, 0], cap[j, 1], cap[j, 2]
                bax, bay, baz = cap[j, 3] - ax_, cap[j, 4] - ay_, cap[j, 5] - az_
                r = cap[j, 6]
                oax, oay, oaz = ox - ax_, oy - ay_, oz - az_
                baba = bax * bax + bay * bay + baz * baz
                baoa = bax * oax + bay * oay + baz * oaz
                s = baoa / baba if baba > 1e-12 else 0.0
                s = min(max(s, 0.0), 1.0)
                px, py, pz = oax - s * bax, oay - s * bay, oaz - s * baz
                if px * px + py * py + pz * pz < r * r:
                    inside = True
                    break
                bard = bax * dx + bay * dy + baz * dz
                rdoa = dx * oax + dy * oay + dz * oaz
                oaoa = oax * oax + oay * oay + oaz * oaz
                A = baba - bard * bard
                B = baba * rdoa - baoa * bard
                C = baba * oaoa - baoa * baoa - r * r * baba
                H = B * B - A * C
                hit = False
                if H >= 0.0 and abs(A) > 1e-12:
                    t = (-B - math.sqrt(H)) / A
                    y = baoa + t * bard
                    if 0.0 < y < baba and 0.0 < t:
                        if t < best:
                            best = t
                        hit = True
                if not hit:
                    for e in range(2):
                        ex = ax_ + (bax if e == 1 else 0.0)
                        ey = ay_ + (bay if e == 1 else 0.0)
                        ez = az_ + (baz if e == 1 else 0.0)
                        qx, qy, qz = ox - ex, oy - ey, oz - ez
                        b = dx * qx + dy * qy + dz * qz
                        c = qx * qx + qy * qy + qz * qz - r * r
                        disc = b * b - c
                        if disc >= 0.0:
                            t = -b - math.sqrt(disc)
                            if 0.0 < t < best:
                                best = t
        out[i] = 0.0 if inside else best
    return out


@njit(cache=True, fastmath=True, inline="always")
def _hit_sphere(ox, oy, oz, dx, dy, dz, cx, cy, cz, r):
    qx, qy, qz = ox - cx, oy - cy, oz - cz
    b = dx * qx + dy * qy + dz * qz
    if b >= 0.0:
        return 1e30
    disc = b * b - (qx * qx + qy * qy + qz * qz - r * r)
    if disc < 0.0:
        return 1e30
    t = -b - math.sqrt(disc)
    return t if t > 0.0 else 1e30


@njit(cache=True, fastmath=True, inline="always")
def _hit_cylinder(ox, oy, oz, dx, dy, dz, cx, cy, r, z0, z1):
    best = 1e30
    qx, qy = ox - cx, oy - cy
    cc = qx * qx + qy * qy - r * r
    a = dx * dx + dy * dy
    if a > 1e-12 and cc > 0.0:
        hb = qx * dx + qy * dy
        disc = hb * hb - a * cc
        if disc >= 0.0:
            t = (-hb - math.sqrt(disc)) / a
            if t > 0.0:
                z = oz + t * dz
                if z0 <= z <= z1:
                    best = t
    if abs(dz) > 1e-12:
        for k in range(2):
            zc = z0 if k == 0 else z1
            t = (zc - oz) / dz
            if 0.0 < t < best:
                hx, hy = qx + t * dx, qy + t * dy
                if hx * hx + hy * hy <= r * r:
                    best = t
    return best


@njit(cache=True, fastmath=True, inline="always")
def _hit_box(ox, oy, oz, dx, dy, dz, bx):
    cyw, syw = math.cos(bx[6]), math.sin(bx[6])
    rx, ry, rz = ox - bx[0], oy - bx[1], oz - bx[2]
    lo0, lo1, lo2 = cyw * rx + syw * ry, -syw * rx + cyw * ry, rz
    ld0, ld1, ld2 = cyw * dx + syw * dy, -syw * dx + cyw * dy, dz
    tmin, tmax = -1e30, 1e30
    for ax in range(3):
        lo = lo0 if ax == 0 else (lo1 if ax == 1 else lo2)
        ld = ld0 if ax == 0 else (ld1 if ax == 1 else ld2)
        h = bx[3 + ax]
        if abs(ld) < 1e-12:
            if lo < -h or lo > h:
                return 1e30
        else:
            t1 = (-h - lo) / ld
            t2 = (h - lo) / ld
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
    if tmax < max(tmin, 0.0) or tmin <= 0.0:
        return 1e30
    return tmin


@njit(cache=True, fastmath=True, inline="always")
def _hit_capsule(ox, oy, oz, dx, dy, dz, cp):
    ax_, ay_, az_ = cp[0], cp[1], cp[2]
    bax, bay, baz = cp[3] - ax_, cp[4] - ay_, cp[5] - az_
    r = cp[6]
    oax, oay, oaz = ox - ax_, oy - ay_, oz - az_
    baba = bax * bax + bay * bay + baz * baz
    baoa = bax * oax + bay * oay + baz * oaz
    bard = bax * dx + bay * dy + baz * dz
    rdoa = dx * oax + dy * oay + dz * oaz
    oaoa = oax * oax + oay * oay + oaz * oaz
    A = baba - bard * bard
    B = baba * rdoa - baoa * bard
    C = baba * oaoa - baoa * baoa - r * r * baba
    H = B * B - A * C
    if H >= 0.0 and abs(A) > 1e-12:
        t = (-B - math.sqrt(H)) / A
        y = baoa + t * bard
        if 0.0 < y < baba and t > 0.0:
            return t
    best = _hit_sphere(ox, oy, oz, dx, dy, dz, ax_, ay_, az_, r)
    t2 = _hit_sphere(ox, oy, oz, dx, dy, dz, ax_ + bax, ay_ + bay, az_ + baz, r)
    return min(best, t2)


@njit(cache=True, fastmath=True)
def raycast_binned(o, dirs, cyl, sph, box, cap, max_range, nbins):
    """
    Same result as `raycast`, but each primitive is only tested against rays whose world azimuth
    falls inside the primitive's angular footprint (as seen from the origin).
    """
    n_rays = dirs.shape[0]
    out = np.full(n_rays, max_range)
    ox, oy, oz = o[0], o[1], o[2]
    pt = np.empty((1, 3))
    pt[0, 0], pt[0, 1], pt[0, 2] = ox, oy, oz
    if distance(pt, cyl, sph, box, cap)[0] <= 0.0:           # origin inside an obstacle
        out[:] = 0.0
        return out
    two_pi = 2.0 * math.pi
    # bucket rays by world azimuth
    ray_bin = np.empty(n_rays, dtype=np.int64)
    counts = np.zeros(nbins + 1, dtype=np.int64)
    for i in range(n_rays):
        a = math.atan2(dirs[i, 1], dirs[i, 0])
        b = int((a + math.pi) / two_pi * nbins) % nbins
        ray_bin[i] = b
        counts[b + 1] += 1
    for b in range(nbins):
        counts[b + 1] += counts[b]
    order = np.empty(n_rays, dtype=np.int64)
    fill = counts[:-1].copy()
    for i in range(n_rays):
        order[fill[ray_bin[i]]] = i
        fill[ray_bin[i]] += 1
    n_prims = sph.shape[0] + cyl.shape[0] + box.shape[0] + cap.shape[0]
    for p in range(n_prims):
        # bounding circle (horizontal) of the primitive
        if p < sph.shape[0]:
            cx, cy, rb = sph[p, 0], sph[p, 1], sph[p, 3]
        elif p < sph.shape[0] + cyl.shape[0]:
            q = p - sph.shape[0]
            cx, cy, rb = cyl[q, 0], cyl[q, 1], cyl[q, 2]
        elif p < sph.shape[0] + cyl.shape[0] + box.shape[0]:
            q = p - sph.shape[0] - cyl.shape[0]
            cx, cy, rb = box[q, 0], box[q, 1], math.sqrt(box[q, 3] ** 2 + box[q, 4] ** 2)
        else:
            q = p - sph.shape[0] - cyl.shape[0] - box.shape[0]
            cx, cy = 0.5 * (cap[q, 0] + cap[q, 3]), 0.5 * (cap[q, 1] + cap[q, 4])
            rb = 0.5 * math.sqrt((cap[q, 3] - cap[q, 0]) ** 2 + (cap[q, 4] - cap[q, 1]) ** 2) + cap[q, 6]
        dxy = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
        if dxy - rb > max_range:
            continue
        if dxy <= rb + 1e-9:
            b_lo, b_hi = 0, nbins - 1
        else:
            half = math.asin(min(1.0, rb / dxy))
            ac = math.atan2(cy - oy, cx - ox)
            b_lo = int(math.floor((ac - half + math.pi) / two_pi * nbins))
            b_hi = int(math.floor((ac + half + math.pi) / two_pi * nbins))
            if b_hi - b_lo >= nbins - 1:
                b_lo, b_hi = 0, nbins - 1
        for bb in range(b_lo, b_hi + 1):
            b = bb % nbins
            for k in range(counts[b], counts[b + 1]):
                i = order[k]
                dx, dy, dz = dirs[i, 0], dirs[i, 1], dirs[i, 2]
                if p < sph.shape[0]:
                    t = _hit_sphere(ox, oy, oz, dx, dy, dz, sph[p, 0], sph[p, 1], sph[p, 2], sph[p, 3])
                elif p < sph.shape[0] + cyl.shape[0]:
                    q = p - sph.shape[0]
                    t = _hit_cylinder(ox, oy, oz, dx, dy, dz, cyl[q, 0], cyl[q, 1], cyl[q, 2], cyl[q, 3], cyl[q, 4])
                elif p < sph.shape[0] + cyl.shape[0] + box.shape[0]:
                    t = _hit_box(ox, oy, oz, dx, dy, dz, box[p - sph.shape[0] - cyl.shape[0]])
                else:
                    t = _hit_capsule(ox, oy, oz, dx, dy, dz, cap[p - sph.shape[0] - cyl.shape[0] - box.shape[0]])
                if t < out[i]:
                    out[i] = t
    return out


@njit(cache=True, fastmath=True)
def distance(points, cyl, sph, box, cap):
    n = points.shape[0]
    out = np.empty(n)
    for i in range(n):
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        best = 1e9
        for j in range(sph.shape[0]):
            d = math.sqrt((px - sph[j, 0]) ** 2 + (py - sph[j, 1]) ** 2 + (pz - sph[j, 2]) ** 2) - sph[j, 3]
            if d < best:
                best = d
        for j in range(cyl.shape[0]):
            dxy = math.sqrt((px - cyl[j, 0]) ** 2 + (py - cyl[j, 1]) ** 2) - cyl[j, 2]
            dz = max(cyl[j, 3] - pz, pz - cyl[j, 4])
            out_d = math.sqrt(max(dxy, 0.0) ** 2 + max(dz, 0.0) ** 2) + min(max(dxy, dz), 0.0)
            if out_d < best:
                best = out_d
        for j in range(box.shape[0]):
            cyw, syw = math.cos(box[j, 6]), math.sin(box[j, 6])
            rx, ry = px - box[j, 0], py - box[j, 1]
            qx = abs(cyw * rx + syw * ry) - box[j, 3]
            qy = abs(-syw * rx + cyw * ry) - box[j, 4]
            qz = abs(pz - box[j, 2]) - box[j, 5]
            d = math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2) + min(max(qx, max(qy, qz)), 0.0)
            if d < best:
                best = d
        for j in range(cap.shape[0]):
            bax, bay, baz = cap[j, 3] - cap[j, 0], cap[j, 4] - cap[j, 1], cap[j, 5] - cap[j, 2]
            pax, pay, paz = px - cap[j, 0], py - cap[j, 1], pz - cap[j, 2]
            baba = bax * bax + bay * bay + baz * baz
            h = (pax * bax + pay * bay + paz * baz) / baba if baba > 1e-12 else 0.0
            h = min(max(h, 0.0), 1.0)
            d = math.sqrt((pax - h * bax) ** 2 + (pay - h * bay) ** 2 + (paz - h * baz) ** 2) - cap[j, 6]
            if d < best:
                best = d
        out[i] = best
    return out


@njit(cache=True)
def rasterize_occupancy(cyl, sph, box, cap, z, inflate, x0, y0, res, nx, ny):
    """Boolean grid (nx, ny): True where a cell centre at height z lies within `inflate` of an obstacle."""
    occ = np.zeros((nx, ny), dtype=np.bool_)
    # spheres and vertical cylinders as discs at height z
    for j in range(sph.shape[0] + cyl.shape[0]):
        if j < sph.shape[0]:
            cx, cy = sph[j, 0], sph[j, 1]
            dzv = abs(z - sph[j, 2])
            rr = sph[j, 3] + inflate
            if dzv >= rr:
                continue
            rad = math.sqrt(rr * rr - dzv * dzv)
        else:
            k = j - sph.shape[0]
            cx, cy = cyl[k, 0], cyl[k, 1]
            if z < cyl[k, 3] - inflate or z > cyl[k, 4] + inflate:
                continue
            rad = cyl[k, 2] + inflate
        i0 = max(int((cx - rad - x0) / res), 0)
        i1 = min(int((cx + rad - x0) / res) + 1, nx - 1)
        j0 = max(int((cy - rad - y0) / res), 0)
        j1 = min(int((cy + rad - y0) / res) + 1, ny - 1)
        for a in range(i0, i1 + 1):
            gx = x0 + (a + 0.5) * res
            for b in range(j0, j1 + 1):
                gy = y0 + (b + 0.5) * res
                if (gx - cx) ** 2 + (gy - cy) ** 2 <= rad * rad:
                    occ[a, b] = True
    # boxes and capsules via the exact distance on cells in their bounding window
    pt = np.empty((1, 3))
    empty5 = np.zeros((0, 5))
    empty4 = np.zeros((0, 4))
    empty7 = np.zeros((0, 7))
    for j in range(box.shape[0] + cap.shape[0]):
        if j < box.shape[0]:
            if abs(z - box[j, 2]) > box[j, 5] + inflate:
                continue
            cx, cy = box[j, 0], box[j, 1]
            rad = math.sqrt(box[j, 3] ** 2 + box[j, 4] ** 2) + inflate
            bsub, ksub = box[j:j + 1], empty7
        else:
            k = j - box.shape[0]
            if z < min(cap[k, 2], cap[k, 5]) - cap[k, 6] - inflate or z > max(cap[k, 2], cap[k, 5]) + cap[k, 6] + inflate:
                continue
            cx, cy = 0.5 * (cap[k, 0] + cap[k, 3]), 0.5 * (cap[k, 1] + cap[k, 4])
            rad = 0.5 * math.sqrt((cap[k, 3] - cap[k, 0]) ** 2 + (cap[k, 4] - cap[k, 1]) ** 2) + cap[k, 6] + inflate
            bsub, ksub = empty7, cap[k:k + 1]
        i0 = max(int((cx - rad - x0) / res), 0)
        i1 = min(int((cx + rad - x0) / res) + 1, nx - 1)
        j0 = max(int((cy - rad - y0) / res), 0)
        j1 = min(int((cy + rad - y0) / res) + 1, ny - 1)
        for a in range(i0, i1 + 1):
            for b in range(j0, j1 + 1):
                if occ[a, b]:
                    continue
                pt[0, 0] = x0 + (a + 0.5) * res
                pt[0, 1] = y0 + (b + 0.5) * res
                pt[0, 2] = z
                if distance(pt, empty5, empty4, bsub, ksub)[0] <= inflate:
                    occ[a, b] = True
    return occ
