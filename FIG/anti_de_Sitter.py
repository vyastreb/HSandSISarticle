#!/usr/bin/env python3
# Draw geodesic LINES of the hyperbolic tiling {p,q} in the Poincaré disk.
# Default {4,5}. Only edges, no fills.
# Requires: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ---------- Hyperbolic utilities ----------
def hdist(z, w):
    num = 2 * abs(z - w)**2
    den = (1 - abs(z)**2) * (1 - abs(w)**2)
    return np.arccosh(1 + num/den)

def invert_in_circle(z, c, R):
    # Euclidean inversion in circle |z-c| = R
    return c + (R**2) * (z - c) / abs(z - c)**2

def circle_from_3pts(a, b, d):
    # Solve for circle through three non-collinear points in C
    ax, ay = a.real, a.imag
    bx, by = b.real, b.imag
    dx, dy = d.real, d.imag
    A = np.array([[ax - bx, ay - by],
                  [bx - dx, by - dy]], dtype=float)
    rhs = 0.5 * np.array([ax*ax + ay*ay - bx*bx - by*by,
                          bx*bx + by*by - dx*dx - dy*dy], dtype=float)
    det = np.linalg.det(A)
    if abs(det) < 1e-14:
        return 0+0j, np.inf
    cx, cy = np.linalg.solve(A, rhs)
    c = cx + 1j*cy
    R = abs(a - c)
    return c, R

def geodesic_circle(z1, z2):
    # Circle orthogonal to unit circle passing through z1,z2
    # Handle diameter case
    if abs(np.imag(z1*np.conj(z2))) < 1e-14 and np.real(z1*np.conj(z2)) > 0:
        return 0+0j, np.inf  # diameter
    z3 = 1 / np.conj(z1)  # inversion of z1 across unit circle
    c, R = circle_from_3pts(z1, z2, z3)
    return c, R

def reflect_across_geodesic(z, z1, z2):
    c, R = geodesic_circle(z1, z2)
    if np.isinf(R):
        ang = np.angle(z1)
        w = z * np.exp(-1j*ang)
        w = np.conj(w)
        return w * np.exp(1j*ang)
    return invert_in_circle(z, c, R)

def geodesic_arc_points(z1, z2, n=100):
    c, R = geodesic_circle(z1, z2)
    if np.isinf(R):
        ang = np.angle(z1)
        r1, r2 = abs(z1), abs(z2)
        rs = np.linspace(r1, r2, n)
        return rs * np.exp(1j*ang)
    a1, a2 = np.angle(z1 - c), np.angle(z2 - c)
    da = (a2 - a1 + np.pi) % (2*np.pi) - np.pi  # shorter arc
    thetas = a1 + np.linspace(0, da, n)
    return c + R * np.exp(1j*thetas)

# ---------- Central regular {p,q} polygon ----------
def central_regular_polygon(p, q):
    # side length ℓ from {p,q}: cosh(ℓ/2) = cos(pi/p)/sin(pi/q)
    ell = 2*np.arccosh(np.cos(np.pi/p)/np.sin(np.pi/q))
    dtheta = 2*np.pi/p

    def d_of_r(r):
        z1 = r * np.exp(1j*0)
        z2 = r * np.exp(1j*dtheta)
        return hdist(z1, z2)

    lo, hi = 1e-6, 0.999999
    for _ in range(80):
        mid = 0.5*(lo + hi)
        if d_of_r(mid) < ell:
            lo = mid
        else:
            hi = mid
    r = 0.5*(lo + hi)
    angles = np.linspace(0, 2*np.pi, p, endpoint=False) + np.pi/p  # rotate for aesthetics
    return [r*np.exp(1j*a) for a in angles]

def poly_edges(poly):
    return [(poly[i], poly[(i+1)%len(poly)]) for i in range(len(poly))]

def reflect_polygon(poly, edge):
    z1, z2 = edge
    return [reflect_across_geodesic(z, z1, z2) for z in poly]

# ---------- Build only edges up to a breadth limit ----------
def tiling_edges(p=4, q=5, generations=5, max_edges=8000):
    seed = central_regular_polygon(p, q)
    Q = deque([(seed, 0)])
    seen_polys = set()  # dedupe by rounded centroid
    edges = set()

    def pkey(poly):
        c = sum(poly)/len(poly)
        return tuple(np.round([c.real, c.imag], 6))

    while Q and len(edges) < max_edges:
        poly, g = Q.popleft()
        key = pkey(poly)
        if key in seen_polys:
            continue
        seen_polys.add(key)

        for e in poly_edges(poly):
            z1, z2 = e
            # store undirected edge id
            key_e = tuple(sorted([tuple(np.round([z1.real, z1.imag], 6)),
                                  tuple(np.round([z2.real, z2.imag], 6))]))
            edges.add(key_e)

            if g < generations:
                child = reflect_polygon(poly, e)
                Q.append((child, g+1))
    return edges

# ---------- Plot ----------
def plot_edges(edge_set, p, q):
    fig, ax = plt.subplots(figsize=(6,6))
    th = np.linspace(0, 2*np.pi, 720)
    ax.plot(np.cos(th), np.sin(th), color='black', lw=1.0)

    for (a, b) in edge_set:
        z1 = a[0] + 1j*a[1]
        z2 = b[0] + 1j*b[1]
        arc = geodesic_arc_points(z1, z2, n=120)
        ax.plot(arc.real, arc.imag, lw=0.6, color='black')

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.01, 1.01); ax.set_ylim(-1.01, 1.01)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("hyperbolic_tiling_edges_p_{}_q_{}.svg".format(p, q))
    plt.show()

if __name__ == "__main__":
    # 1/p+1/q < 1/2

    # p = 4
    # q = 5
    # edges = tiling_edges(p=p, q=q, generations=6, max_edges=120)
    # plot_edges(edges, p=p, q=q)

    p = 7
    q = 3
    edges = tiling_edges(p=p, q=q, generations=7, max_edges=3000)
    plot_edges(edges, p=p, q=q)

    # p = 5
    # q = 4
    # edges = tiling_edges(p=p, q=q, generations=6, max_edges=120)
    # plot_edges(edges, p=p, q=q)

    # p = 6
    # q = 4
    # edges = tiling_edges(p=p, q=q, generations=15, max_edges=10000)
    # plot_edges(edges, p=p, q=q)
