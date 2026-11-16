"""Skeleton/graph geometric trajectory-recovery baseline with a right-to-left (RTL) prior.

Rule (documented in ouhdl_runs_methods.md):
 1. Binarise the input image (ink dark on light; Otsu, then remove CCs < min_area px).
 2. Skeletonise (scikit-image ``skeletonize``).
 3. Build the 8-connected skeleton pixel graph.  Nodes = endpoints (degree 1) and junction
    clusters (degree >= 3, 8-connected junction pixels merged); edges = pixel chains between
    nodes ("cut at junctions").  Isolated loops become one closed edge.
 4. Each skeleton connected component is a *body*.  A body is a *dot/diacritic* if the
    bounding box of its ink is smaller than ``dot_frac`` x unit height in both directions;
    otherwise it is a *main body*.
 5. Main bodies are visited right-to-left by their rightmost x.  Inside a body the walk starts
    at the rightmost endpoint (ties -> topmost); if the body has no endpoint, at its
    rightmost-topmost node.  At every node the walk continues along the unvisited edge with
    the smallest turning angle (first edge of a stroke: the most leftward-heading edge);
    when no unvisited edge is incident the pen is lifted and a new stroke starts at the
    rightmost(-topmost) node that still has unvisited edges (endpoints preferred).
 6. Dots are emitted after the main bodies, right-to-left by x-centre (``dots='last'``), or
    right after the main body nearest to them (``dots='after_body'``, line-level variant).
Output strokes are in input-image pixel coordinates (x right, y down).
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def binarise(img: np.ndarray, min_area: int = 3) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = g.astype(np.uint8)
    if g.max() == g.min():
        return np.zeros_like(g, bool)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b = b > 0
    lab, n = ndi.label(b, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(b, lab, index=np.arange(1, n + 1))
        keep = np.r_[False, sizes >= min_area]
        b = keep[lab]
    return b


def _skeleton_graph(sk: np.ndarray):
    ys, xs = np.nonzero(sk)
    pix = set(zip(ys.tolist(), xs.tolist()))
    nbrs = {p: [(p[0] + dy, p[1] + dx) for dy, dx in _NB if (p[0] + dy, p[1] + dx) in pix] for p in pix}
    deg = {p: len(v) for p, v in nbrs.items()}
    junction = {p for p, d in deg.items() if d >= 3}
    jid, cl = {}, 0
    for p in junction:
        if p in jid:
            continue
        stack = [p]
        jid[p] = cl
        while stack:
            q = stack.pop()
            for r in nbrs[q]:
                if r in junction and r not in jid:
                    jid[r] = cl
                    stack.append(r)
        cl += 1
    node_of = {}
    for p, d in deg.items():
        if p in jid:
            node_of[p] = ("j", jid[p])
        elif d <= 1:
            node_of[p] = ("e", p)
    edges, used = [], set()
    for p in list(node_of):
        for r in nbrs[p]:
            key = frozenset([p, r])
            if key in used:
                continue
            if r in node_of:
                used.add(key)
                if node_of[r] != node_of[p]:
                    edges.append((node_of[p], node_of[r], [p, r]))
                continue
            path = [p, r]
            used.add(key)
            prev, cur = p, r
            while cur not in node_of:
                nxt = [q for q in nbrs[cur] if q != prev and frozenset([cur, q]) not in used]
                if not nxt:
                    break
                nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
                q = nxt[0]
                used.add(frozenset([cur, q]))
                path.append(q)
                prev, cur = cur, q
            end = node_of.get(cur, ("e", cur))
            edges.append((node_of[p], end, path))
    seen = {q for e in edges for q in e[2]}
    for p in pix:
        if p in seen or p in node_of:
            continue
        loop, prev, cur = [p], None, p
        seen.add(p)
        while True:
            nxt = [q for q in nbrs[cur] if q != prev and q not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        loop.append(p)
        edges.append((("l", p), ("l", p), loop))
    return pix, node_of, edges


def _xy(path):
    return np.array([[c, r] for r, c in path], dtype=np.float64)


def _walk_body(edge_ids, edges, node_pos):
    inc = defaultdict(list)
    for k in edge_ids:
        a, b, _ = edges[k]
        inc[a].append(k)
        if b != a:
            inc[b].append(k)
    unvisited = set(edge_ids)

    def start_node():
        cand = [n for n in inc if any(k in unvisited for k in inc[n])]
        ends = [n for n in cand if n[0] == "e" and sum(1 for k in inc[n] if k in unvisited) == 1]
        pool = ends if ends else cand
        return max(pool, key=lambda n: (round(node_pos[n][0]), -node_pos[n][1]))

    def oriented(k, n):
        a, b, p = edges[k]
        xy = _xy(p)
        if a == b:
            if a[0] == "l":  # free loop: start at its rightmost-topmost pixel
                i = int(np.lexsort((xy[:-1, 1], -xy[:-1, 0]))[0])
                xy = np.vstack([xy[i:-1], xy[:i + 1]])
            return xy, b
        return (xy, b) if a == n else (xy[::-1], a)

    strokes = []
    while unvisited:
        n = start_node()
        pts, heading = [], None
        while True:
            opts = [k for k in inc[n] if k in unvisited]
            if not opts:
                break
            best, best_cost = None, None
            for k in opts:
                xy, other = oriented(k, n)
                v = xy[min(len(xy) - 1, 5)] - xy[0]
                ref = np.array([-1.0, 0.0]) if heading is None else heading
                cost = -np.dot(v, ref) / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9)
                if best_cost is None or cost < best_cost:
                    best, best_cost = (k, xy, other), cost
            k, xy, other = best
            unvisited.discard(k)
            pts.append(xy if not pts else xy[1:])
            tail = xy[max(0, len(xy) - 6):]
            if len(tail) > 1 and np.linalg.norm(tail[-1] - tail[0]) > 0:
                heading = tail[-1] - tail[0]
            n = other
        if pts:
            strokes.append(np.vstack(pts))
    return strokes


def recover(img: np.ndarray, dot_frac: float = 0.18, dots: str = "last",
            min_area: int = 3, unit_height: Optional[float] = None) -> List[np.ndarray]:
    """Recover an ordered trajectory (list of (n,2) xy strokes, image px) from an image with
    dark ink on a light background.  ``unit_height``: reference height for the dot test
    (default: image height)."""
    b = binarise(img, min_area=min_area)
    if not b.any():
        return []
    sk = skeletonize(b)
    pix, node_of, edges = _skeleton_graph(sk)
    node_pix = defaultdict(list)
    for p, n in node_of.items():
        node_pix[n].append(p)
    for a, bn, path in edges:
        for n, p in ((a, path[0]), (bn, path[-1])):
            if n not in node_pix:
                node_pix[n].append(p)
    node_pos = {n: np.mean([[c, r] for r, c in ps], axis=0) for n, ps in node_pix.items()}
    lab, nb = ndi.label(sk, structure=np.ones((3, 3)))
    body_edges = defaultdict(list)
    for k, (_, _, path) in enumerate(edges):
        body_edges[int(lab[path[0]])].append(k)
    H = float(unit_height or img.shape[0])
    ink_lab, _ = ndi.label(b, structure=np.ones((3, 3)))
    objs = ndi.find_objects(ink_lab)
    bodies = []
    for bid in range(1, nb + 1):
        ys, xs = np.nonzero(lab == bid)
        il = int(ink_lab[ys[0], xs[0]])
        sl = objs[il - 1]
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        is_dot = (w < dot_frac * H) and (h < dot_frac * H)
        strokes = _walk_body(body_edges[bid], edges, node_pos) if body_edges[bid] else \
            [np.array([[xs.mean(), ys.mean()]])]
        bodies.append(dict(strokes=strokes, xmax=float(xs.max()), xc=float(xs.mean()),
                           yc=float(ys.mean()), dot=is_dot, pts=np.c_[xs, ys].astype(float)))
    mains = sorted([q for q in bodies if not q["dot"]], key=lambda q: -q["xmax"])
    dotsl = sorted([q for q in bodies if q["dot"]], key=lambda q: -q["xc"])
    out = []
    if dots == "last" or not mains:
        for q in mains + dotsl:
            out.extend(q["strokes"])
        return out
    trees = [cKDTree(q["pts"]) for q in mains]
    attach = defaultdict(list)
    for d in dotsl:
        dist = [t.query([d["xc"], d["yc"]])[0] for t in trees]
        attach[int(np.argmin(dist))].append(d)
    for i, q in enumerate(mains):
        out.extend(q["strokes"])
        for d in sorted(attach[i], key=lambda z: -z["xc"]):
            out.extend(d["strokes"])
    return out
