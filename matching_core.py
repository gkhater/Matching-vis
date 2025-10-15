"""Standalone maximum-weight matching with warm-start support.

This module contains a self-contained implementation of Edmonds' blossom
algorithm adapted from the project codebase.  It exposes a single function
``max_weight_matching`` that accepts optional warm-start matchings and dual
variables, making it suitable for reuse in other projects.
For backward compatibility with the visualization pipeline, the helper
``export_matching_run`` writes history snapshots using the original schema
expected by :mod:`visualize_evolution`.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEBUG = None
CHECK_DELTA = False
CHECK_OPTIMUM = True


def max_weight_matching(
    edges: Iterable[Tuple[int, int, float]],
    *,
    max_cardinality: bool = False,
    initial_mate: Optional[Sequence[int]] = None,
    initial_dualvar: Optional[Sequence[float]] = None,
    initial_state: Optional[dict] = None,
    return_state: bool = False,
    max_stages: Optional[int] = None,
):
    """Compute a maximum-weight matching using Edmonds' blossom algorithm."""

    edges = list(edges)

    from sys import version as sys_version
    if sys_version < '3':
        integer_types = (int, long)  # type: ignore[name-defined]
    else:
        integer_types = (int,)

    if not edges:
        return ([], [], [], [], []) if not return_state else ([], [], [], [], [], None)

    nedge = len(edges)
    nvertex = 0
    for (i, j, w) in edges:
        if i < 0 or j < 0 or i == j:
            raise ValueError("edges must connect distinct non-negative vertices")
        if i >= nvertex:
            nvertex = i + 1
        if j >= nvertex:
            nvertex = j + 1

    maxweight = max(0, max([wt for (_, _, wt) in edges])) if edges else 0

    endpoint = [edges[p // 2][p % 2] for p in range(2 * nedge)]

    neighbend = [[] for _ in range(nvertex)]
    for k, (i, j, _) in enumerate(edges):
        neighbend[i].append(2 * k + 1)
        neighbend[j].append(2 * k)

    stage = 0
    blossom_history: List[dict] = []
    matching_history: List[List[int]] = []
    tight_edges_history: List[List[Tuple[int, int]]] = []

    def initialize_default_state():
        mate_local = nvertex * [-1]
        if initial_mate is not None:
            edge_map = {}
            for k, (i, j, _) in enumerate(edges):
                edge_map[(i, j)] = k
                edge_map[(j, i)] = k

            for i, j in enumerate(initial_mate):
                if j != -1 and i < j:
                    k = edge_map.get((i, j))
                    if k is not None:
                        v_edge, w_edge, _ = edges[k]
                        if v_edge == i:
                            mate_local[i] = 2 * k + 1
                            mate_local[j] = 2 * k
                        else:
                            mate_local[j] = 2 * k + 1
                            mate_local[i] = 2 * k

        label_local = (2 * nvertex) * [0]
        labelend_local = (2 * nvertex) * [-1]
        inblossom_local = list(range(nvertex))
        blossomparent_local = (2 * nvertex) * [-1]
        blossomchilds_local = (2 * nvertex) * [None]
        blossombase_local = list(range(nvertex)) + nvertex * [-1]
        blossomendps_local = (2 * nvertex) * [None]
        bestedge_local = (2 * nvertex) * [-1]
        blossombestedges_local = (2 * nvertex) * [None]
        unusedblossoms_local = list(range(nvertex, 2 * nvertex))
        if initial_dualvar is not None:
            dualvar_local = list(initial_dualvar)
        else:
            dualvar_local = nvertex * [maxweight] + nvertex * [0]
        allowedge_local = nedge * [False]
        queue_local: List[int] = []
        return (
            mate_local,
            label_local,
            labelend_local,
            inblossom_local,
            blossomparent_local,
            blossomchilds_local,
            blossombase_local,
            blossomendps_local,
            bestedge_local,
            blossombestedges_local,
            unusedblossoms_local,
            dualvar_local,
            allowedge_local,
            queue_local,
        )

    def clone_state_from_snapshot(snapshot):
        required_keys = [
            'mate', 'label', 'labelend', 'inblossom', 'blossomparent',
            'blossomchilds', 'blossombase', 'blossomendps', 'bestedge',
            'blossombestedges', 'unusedblossoms', 'dualvar', 'allowedge',
            'queue', 'stage',
        ]
        for key in required_keys:
            if key not in snapshot:
                raise ValueError(f"initial_state missing required key '{key}'")

        mate_local = snapshot['mate'][:]
        label_local = snapshot['label'][:]
        labelend_local = snapshot['labelend'][:]
        inblossom_local = snapshot['inblossom'][:]
        blossomparent_local = snapshot['blossomparent'][:]
        blossomchilds_local = copy.deepcopy(snapshot['blossomchilds'])
        blossombase_local = snapshot['blossombase'][:]
        blossomendps_local = copy.deepcopy(snapshot['blossomendps'])
        bestedge_local = snapshot['bestedge'][:]
        blossombestedges_local = copy.deepcopy(snapshot['blossombestedges'])
        unusedblossoms_local = snapshot['unusedblossoms'][:]
        dualvar_local = snapshot['dualvar'][:]
        allowedge_local = snapshot['allowedge'][:]
        queue_local = snapshot['queue'][:]

        return (
            mate_local,
            label_local,
            labelend_local,
            inblossom_local,
            blossomparent_local,
            blossomchilds_local,
            blossombase_local,
            blossomendps_local,
            bestedge_local,
            blossombestedges_local,
            unusedblossoms_local,
            dualvar_local,
            allowedge_local,
            queue_local,
        ), snapshot.get('stage', 0), copy.deepcopy(snapshot.get('blossom_history', [])), copy.deepcopy(snapshot.get('matching_history', [])), copy.deepcopy(snapshot.get('tight_edges_history', []))

    def validate_state_consistency():
        if len(mate) != nvertex:
            raise ValueError("State mate length mismatch")
        if len(label) != 2 * nvertex:
            raise ValueError("State label length mismatch")
        if len(labelend) != 2 * nvertex:
            raise ValueError("State labelend length mismatch")
        if len(inblossom) != nvertex:
            raise ValueError("State inblossom length mismatch")
        if len(blossomparent) != 2 * nvertex:
            raise ValueError("State blossomparent length mismatch")
        if len(blossomchilds) != 2 * nvertex:
            raise ValueError("State blossomchilds length mismatch")
        if len(blossombase) != 2 * nvertex:
            raise ValueError("State blossombase length mismatch")
        if len(blossomendps) != 2 * nvertex:
            raise ValueError("State blossomendps length mismatch")
        if len(bestedge) != 2 * nvertex:
            raise ValueError("State bestedge length mismatch")
        if len(blossombestedges) != 2 * nvertex:
            raise ValueError("State blossombestedges length mismatch")
        if len(dualvar) != 2 * nvertex:
            raise ValueError("State dualvar length mismatch")
        if len(allowedge) != nedge:
            raise ValueError("State allowedge length mismatch")
        if stage < 0 or stage > nvertex:
            raise ValueError("State stage out of range")

    if initial_state is not None:
        (
            mate,
            label,
            labelend,
            inblossom,
            blossomparent,
            blossomchilds,
            blossombase,
            blossomendps,
            bestedge,
            blossombestedges,
            unusedblossoms,
            dualvar,
            allowedge,
            queue,
        ), stage, blossom_history, matching_history, tight_edges_history = clone_state_from_snapshot(initial_state)
        if initial_mate is not None or initial_dualvar is not None:
            raise ValueError("Provide either initial_state or initial_mate/initial_dualvar, not both")
    else:
        (
            mate,
            label,
            labelend,
            inblossom,
            blossomparent,
            blossomchilds,
            blossombase,
            blossomendps,
            bestedge,
            blossombestedges,
            unusedblossoms,
            dualvar,
            allowedge,
            queue,
        ) = initialize_default_state()
        stage = 0
    validate_state_consistency()

    def slack(k):
        (i, j, wt) = edges[k]
        return dualvar[i] + dualvar[j] - 2 * wt

    def blossomLeaves(b):
        if b < nvertex:
            yield b
        else:
            for t in blossomchilds[b]:
                if t < nvertex:
                    yield t
                else:
                    for v in blossomLeaves(t):
                        yield v

    def assignLabel(w, t, p):
        if DEBUG: DEBUG('assignLabel(%d,%d,%d)' % (w, t, p))
        b = inblossom[w]
        assert label[w] == 0 and label[b] == 0
        label[w] = label[b] = t
        labelend[w] = labelend[b] = p
        bestedge[w] = bestedge[b] = -1
        if t == 1:
            queue.extend(list(blossomLeaves(b)))
        elif t == 2:
            base = blossombase[b]
            assert mate[base] >= 0
            assignLabel(endpoint[mate[base]], 1, mate[base] ^ 1)

    def scanBlossom(v, w):
        if DEBUG: DEBUG('scanBlossom(%d,%d)' % (v, w))
        path = []
        base = -1
        while v != -1 or w != -1:
            b = inblossom[v]
            if label[b] & 4:
                base = blossombase[b]
                break
            assert label[b] == 1
            path.append(b)
            label[b] = 5
            assert labelend[b] == mate[blossombase[b]]
            if labelend[b] == -1:
                v = -1
            else:
                v = endpoint[labelend[b]]
                b = inblossom[v]
                assert label[b] == 2
                assert labelend[b] >= 0
                v = endpoint[labelend[b]]
            if w != -1:
                v, w = w, v
        for b in path:
            label[b] = 1
        return base

    def addBlossom(base, k, t):
        (v, w, wt) = edges[k]
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        b = unusedblossoms.pop()
        if DEBUG: DEBUG('addBlossom(%d,%d) (v=%d w=%d) -> %d' % (base, k, v, w, b))
        blossombase[b] = base
        blossomparent[b] = -1
        blossomparent[bb] = b
        blossomchilds[b] = path = []
        blossomendps[b] = endps = []
        while bv != bb:
            blossomparent[bv] = b
            path.append(bv)
            endps.append(labelend[bv])
            assert labelend[bv] >= 0
            v = endpoint[labelend[bv]]
            bv = inblossom[v]
        path.append(bb)
        path.reverse()
        endps.reverse()
        endps.append(2 * k)
        while bw != bb:
            blossomparent[bw] = b
            path.append(bw)
            endps.append(labelend[bw] ^ 1)
            assert labelend[bw] >= 0
            w = endpoint[labelend[bw]]
            bw = inblossom[w]
        assert label[bb] == 1
        label[b] = 1
        labelend[b] = labelend[bb]
        dualvar[b] = 0

        for v in blossomLeaves(b):
            if label[inblossom[v]] == 2:
                queue.append(v)
            inblossom[v] = b
        bestedgeto = (2 * nvertex) * [-1]
        for bv in path:
            if blossombestedges[bv] is None:
                nblists = [[p // 2 for p in neighbend[v]] for v in blossomLeaves(bv)]
            else:
                nblists = [blossombestedges[bv]]
            for nblist in nblists:
                for k in nblist:
                    (i, j, wt) = edges[k]
                    if inblossom[j] == b:
                        i, j = j, i
                    bj = inblossom[j]
                    if bj != b and label[bj] == 1 and (
                        bestedgeto[bj] == -1 or slack(k) < slack(bestedgeto[bj])
                    ):
                        bestedgeto[bj] = k
            blossombestedges[bv] = None
            bestedge[bv] = -1
        blossombestedges[b] = [k for k in bestedgeto if k != -1]
        bestedge[b] = -1
        for k in blossombestedges[b]:
            if bestedge[b] == -1 or slack(k) < slack(bestedge[b]):
                bestedge[b] = k

        blossom_history.append(
            {
                'blossom_id': b,
                'base': base,
                'leaves': list(blossomLeaves(b)),
                'size': len(list(blossomLeaves(b))),
                'stage': t,
            }
        )

    def expandBlossom(b, endstage):
        for s in blossomchilds[b]:
            blossomparent[s] = -1
            if s < nvertex:
                inblossom[s] = s
            elif endstage and dualvar[s] == 0:
                expandBlossom(s, endstage)
            else:
                for v in blossomLeaves(s):
                    inblossom[v] = s
        if (not endstage) and label[b] == 2:
            assert labelend[b] >= 0
            entrychild = inblossom[endpoint[labelend[b] ^ 1]]
            j = blossomchilds[b].index(entrychild)
            if j & 1:
                j -= len(blossomchilds[b])
                jstep = 1
                endptrick = 0
            else:
                jstep = -1
                endptrick = 1
            p = labelend[b]
            while j != 0:
                label[endpoint[p ^ 1]] = 0
                label[endpoint[blossomendps[b][j - endptrick] ^ endptrick ^ 1]] = 0
                assignLabel(endpoint[p ^ 1], 2, p)
                allowedge[blossomendps[b][j - endptrick] // 2] = True
                j += jstep
                p = blossomendps[b][j - endptrick] ^ endptrick
                allowedge[p // 2] = True
                j += jstep
            bv = blossomchilds[b][j]
            label[endpoint[p ^ 1]] = label[bv] = 2
            labelend[endpoint[p ^ 1]] = labelend[bv] = p
            bestedge[bv] = -1
            j += jstep
            while blossomchilds[b][j] != entrychild:
                bv = blossomchilds[b][j]
                if label[bv] == 1:
                    j += jstep
                    continue
                for v in blossomLeaves(bv):
                    if label[v] != 0:
                        break
                if label[v] != 0:
                    label[v] = 0
                    label[endpoint[mate[blossombase[bv]]]] = 0
                    assignLabel(v, 2, labelend[v])
                j += jstep
        label[b] = labelend[b] = -1
        blossomchilds[b] = blossomendps[b] = None
        blossombase[b] = -1
        blossombestedges[b] = None
        bestedge[b] = -1
        unusedblossoms.append(b)

    def augmentBlossom(b, v):
        t = v
        while blossomparent[t] != b:
            t = blossomparent[t]
        if t >= nvertex:
            augmentBlossom(t, v)
        i = j = blossomchilds[b].index(t)
        if i & 1:
            j -= len(blossomchilds[b])
            jstep = 1
            endptrick = 0
        else:
            jstep = -1
            endptrick = 1
        while j != 0:
            j += jstep
            t = blossomchilds[b][j]
            p = blossomendps[b][j - endptrick] ^ endptrick
            if t >= nvertex:
                augmentBlossom(t, endpoint[p])
            j += jstep
            t = blossomchilds[b][j]
            if t >= nvertex:
                augmentBlossom(t, endpoint[p ^ 1])
            mate[endpoint[p]] = p ^ 1
            mate[endpoint[p ^ 1]] = p
        blossomchilds[b] = blossomchilds[b][i:] + blossomchilds[b][:i]
        blossomendps[b] = blossomendps[b][i:] + blossomendps[b][:i]
        blossombase[b] = blossombase[blossomchilds[b][0]]

    def augmentMatching(k):
        (v, w, wt) = edges[k]
        if DEBUG: DEBUG('augmentMatching(%d) (v=%d w=%d)' % (k, v, w))
        for (s, p) in ((v, 2 * k + 1), (w, 2 * k)):
            while True:
                bs = inblossom[s]
                assert label[bs] == 1
                assert labelend[bs] == mate[blossombase[bs]]
                if bs >= nvertex:
                    augmentBlossom(bs, s)
                mate[s] = p
                if labelend[bs] == -1:
                    break
                t = endpoint[labelend[bs]]
                bt = inblossom[t]
                assert label[bt] == 2
                assert labelend[bt] >= 0
                s = endpoint[labelend[bt]]
                j = endpoint[labelend[bt] ^ 1]
                if bt >= nvertex:
                    augmentBlossom(bt, j)
                mate[j] = labelend[bt]
                p = labelend[bt] ^ 1

    def verifyOptimum():
        if max_cardinality:
            vdualoffset = max(0, -min(dualvar[:nvertex]))
        else:
            vdualoffset = 0
        assert min(dualvar[:nvertex]) + vdualoffset >= 0
        assert min(dualvar[nvertex:]) >= 0
        for k in range(nedge):
            (i, j, wt) = edges[k]
            s = dualvar[i] + dualvar[j] - 2 * wt
            iblossoms = [i]
            jblossoms = [j]
            while blossomparent[iblossoms[-1]] != -1:
                iblossoms.append(blossomparent[iblossoms[-1]])
            while blossomparent[jblossoms[-1]] != -1:
                jblossoms.append(blossomparent[jblossoms[-1]])
            iblossoms.reverse()
            jblossoms.reverse()
            for (bi, bj) in zip(iblossoms, jblossoms):
                if bi != bj:
                    break
                s += 2 * dualvar[bi]
            assert s >= 0
            if mate[i] // 2 == k or mate[j] // 2 == k:
                assert mate[i] // 2 == k and mate[j] // 2 == k
                assert s == 0
        for v in range(nvertex):
            assert mate[v] >= 0 or dualvar[v] + vdualoffset == 0

    for t in range(stage, nvertex):
        label[:] = (2 * nvertex) * [0]
        bestedge[:] = (2 * nvertex) * [-1]
        blossombestedges[nvertex:] = nvertex * [None]
        allowedge[:] = nedge * [False]
        queue[:] = []
        for v in range(nvertex):
            if mate[v] == -1 and label[inblossom[v]] == 0:
                assignLabel(v, 1, -1)
        augmented = 0
        while True:
            while queue and not augmented:
                v = queue.pop()
                assert label[inblossom[v]] == 1
                for p in neighbend[v]:
                    k = p // 2
                    w = endpoint[p]
                    if inblossom[v] == inblossom[w]:
                        continue
                    if not allowedge[k]:
                        kslack = slack(k)
                        if kslack <= 0:
                            allowedge[k] = True
                    if allowedge[k]:
                        if label[inblossom[w]] == 0:
                            assignLabel(w, 2, p ^ 1)
                        elif label[inblossom[w]] == 1:
                            base = scanBlossom(v, w)
                            if base >= 0:
                                addBlossom(base, k, t)
                            else:
                                augmentMatching(k)
                                augmented = 1
                                break
                        elif label[w] == 0:
                            assert label[inblossom[w]] == 2
                            label[w] = 2
                            labelend[w] = p ^ 1
                    elif label[inblossom[w]] == 1:
                        b = inblossom[v]
                        if bestedge[b] == -1 or kslack < slack(bestedge[b]):
                            bestedge[b] = k
                    elif label[w] == 0:
                        if bestedge[w] == -1 or kslack < slack(bestedge[w]):
                            bestedge[w] = k
            if augmented:
                break
            deltatype = -1
            delta = deltaedge = deltablossom = None
            if not max_cardinality:
                deltatype = 1
                delta = min(dualvar[:nvertex])
            for v in range(nvertex):
                if label[inblossom[v]] == 0 and bestedge[v] != -1:
                    d = slack(bestedge[v])
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 2
                        deltaedge = bestedge[v]
            for b in range(2 * nvertex):
                if blossomparent[b] == -1 and label[b] == 1 and bestedge[b] != -1:
                    kslack = slack(bestedge[b])
                    if isinstance(kslack, integer_types):
                        assert (kslack % 2) == 0
                        d = kslack // 2
                    else:
                        d = kslack / 2
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 3
                        deltaedge = bestedge[b]
            for b in range(nvertex, 2 * nvertex):
                if (
                    blossombase[b] >= 0
                    and blossomparent[b] == -1
                    and label[b] == 2
                    and (deltatype == -1 or dualvar[b] < delta)
                ):
                    delta = dualvar[b]
                    deltatype = 4
                    deltablossom = b
            if deltatype == -1:
                assert max_cardinality
                deltatype = 1
                delta = max(0, min(dualvar[:nvertex]))
            for v in range(nvertex):
                if label[inblossom[v]] == 1:
                    dualvar[v] -= delta
                elif label[inblossom[v]] == 2:
                    dualvar[v] += delta
            for b in range(nvertex, 2 * nvertex):
                if blossombase[b] >= 0 and blossomparent[b] == -1:
                    if label[b] == 1:
                        dualvar[b] += delta
                    elif label[b] == 2:
                        dualvar[b] -= delta
            if deltatype == 1:
                break
            elif deltatype == 2:
                allowedge[deltaedge] = True
                (i, j, wt) = edges[deltaedge]
                if label[inblossom[i]] == 0:
                    i, j = j, i
                queue.append(i)
            elif deltatype == 3:
                allowedge[deltaedge] = True
                (i, j, wt) = edges[deltaedge]
                queue.append(i)
            elif deltatype == 4:
                expandBlossom(deltablossom, False)
        stage = t + 1
        if not augmented:
            break
        for b in range(nvertex, 2 * nvertex):
            if blossomparent[b] == -1 and blossombase[b] >= 0 and label[b] == 1 and dualvar[b] == 0:
                expandBlossom(b, True)
        mate_for_history = [-1] * nvertex
        for i in range(nvertex):
            if mate[i] != -1:
                mate_for_history[i] = endpoint[mate[i]]
        matching_history.append(mate_for_history)

        current_tight_edges = []
        for k in range(nedge):
            if slack(k) == 0:
                (i, j, w) = edges[k]
                current_tight_edges.append((i, j))
        tight_edges_history.append(current_tight_edges)
        if max_stages is not None and stage >= max_stages:
            break
    if CHECK_OPTIMUM and stage >= nvertex:
        verifyOptimum()

    def build_state_snapshot():
        return {
            'mate': mate[:],
            'label': label[:],
            'labelend': labelend[:],
            'inblossom': inblossom[:],
            'blossomparent': blossomparent[:],
            'blossomchilds': copy.deepcopy(blossomchilds),
            'blossombase': blossombase[:],
            'blossomendps': copy.deepcopy(blossomendps),
            'bestedge': bestedge[:],
            'blossombestedges': copy.deepcopy(blossombestedges),
            'unusedblossoms': unusedblossoms[:],
            'dualvar': dualvar[:],
            'allowedge': allowedge[:],
            'queue': queue[:],
            'stage': stage,
            'blossom_history': copy.deepcopy(blossom_history),
            'matching_history': copy.deepcopy(matching_history),
            'tight_edges_history': copy.deepcopy(tight_edges_history),
        }

    state_snapshot = build_state_snapshot() if return_state else None

    for v in range(nvertex):
        if mate[v] >= 0:
            mate[v] = endpoint[mate[v]]
    for v in range(nvertex):
        assert mate[v] == -1 or mate[mate[v]] == v

    result = (mate, blossom_history, matching_history, tight_edges_history, dualvar)
    if return_state:
        result = result + (state_snapshot,)
    return result


def compute_matching_map(
    edges: Iterable[Tuple[int, int, float]],
    *,
    max_cardinality: bool = False,
    initial_mate: Optional[Sequence[int]] = None,
    initial_dualvar: Optional[Sequence[float]] = None,
) -> Dict[int, int]:
    """Convenience wrapper returning a vertex→partner dictionary."""

    mate, *_ = max_weight_matching(
        edges,
        max_cardinality=max_cardinality,
        initial_mate=initial_mate,
        initial_dualvar=initial_dualvar,
    )

    matching: Dict[int, int] = {}
    for i, partner in enumerate(mate):
        if partner != -1 and i < partner:
            matching[i] = partner
            matching[partner] = i
    return matching


def export_matching_run(
    *,
    run_id: str,
    graph_metadata: Dict[str, object],
    node_labels: Dict[int, str],
    edges: Iterable[Tuple[int, int, float]],
    max_cardinality: bool = False,
    initial_mate: Optional[Sequence[int]] = None,
    initial_dualvar: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Run the matcher and package results for visualization."""

    mate, blossom_history, matching_history, tight_edges_history, dualvar = max_weight_matching(
        edges,
        max_cardinality=max_cardinality,
        initial_mate=initial_mate,
        initial_dualvar=initial_dualvar,
    )

    processed_matching_history = []
    for mate_snapshot in matching_history:
        stage_data: Dict[str, object] = {
            'paths': [],
            'path_endpoints': [],
            'unmatched_defects': [],
        }
        seen = set()
        for i, partner in enumerate(mate_snapshot):
            if partner != -1 and i < partner:
                stage_data['paths'].append([node_labels.get(i, str(i)), node_labels.get(partner, str(partner))])
                seen.add(i)
                seen.add(partner)
        for i, partner in enumerate(mate_snapshot):
            if partner == -1:
                stage_data['path_endpoints'].append([node_labels.get(i, str(i))])
        processed_matching_history.append(stage_data)

    processed_tight_edges = []
    for stage_edges in tight_edges_history:
        processed_tight_edges.append([
        [node_labels.get(i, str(i)), node_labels.get(j, str(j))] for (i, j) in stage_edges
        ])

    data = {
        'run_id': run_id,
        'int_to_node': {str(idx): label for idx, label in node_labels.items()},
        'edges': [(int(u), int(v), float(w)) for (u, v, w) in edges],
        'mate': mate,
        'dual_variables': dualvar,
        'blossom_history': blossom_history,
        'matching_history': processed_matching_history,
        'tight_edges_history': processed_tight_edges,
    }
    data.update(graph_metadata)
    return data


def save_matching_run(
    json_path: str,
    run_data: Dict[str, object],
    *,
    append: bool = True,
) -> None:
    """Persist run data to a JSON file compatible with ``visualize_evolution``."""

    runs: List[Dict[str, object]] = []
    if append and os.path.exists(json_path):
        with open(json_path, 'r') as fh:
            try:
                existing = json.load(fh)
                if isinstance(existing, list):
                    runs = existing
                elif existing:
                    runs = [existing]
            except json.JSONDecodeError:
                runs = []
    else:
        runs = []

    runs.append(run_data)

    with open(json_path, 'w') as fh:
        json.dump(runs, fh, indent=2)


__all__ = [
    "max_weight_matching",
    "compute_matching_map",
    "export_matching_run",
    "save_matching_run",
]
