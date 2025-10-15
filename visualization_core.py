"""General-purpose visualization helpers for blossom matching runs.

The helpers provided here operate on the JSON-compatible dictionaries produced
by :mod:`matching_core`.  They do not assume any lattice- or gadget-specific
structure; instead they rely solely on the metadata embedded in each run.  If
node positions are not provided, a spring layout is computed automatically.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def _extract_label_mapping(run_data: Dict[str, object]) -> Dict[int, str]:
    mapping_raw = run_data.get('int_to_node', {}) or {}
    return {int(k): str(v) for k, v in mapping_raw.items()}


def _label_for(idx: int, labels: Dict[int, str]) -> str:
    return labels.get(idx, str(idx))


def _build_graph(run_data: Dict[str, object], labels: Dict[int, str]) -> nx.Graph:
    graph = nx.Graph()
    for entry in run_data.get('edges', []):
        if len(entry) < 2:
            continue
        u_idx, v_idx = entry[0], entry[1]
        weight = entry[2] if len(entry) > 2 else 1.0
        u_label = _label_for(int(u_idx), labels)
        v_label = _label_for(int(v_idx), labels)
        graph.add_edge(u_label, v_label, weight=weight)
    for label in (_label_for(k, labels) for k in labels):
        if label not in graph:
            graph.add_node(label)
    return graph


def _resolve_positions(
    graph: nx.Graph,
    run_data: Dict[str, object],
    labels: Dict[int, str],
) -> Dict[str, Tuple[float, float]]:
    raw_positions = run_data.get('node_positions', None)
    positions: Dict[str, Tuple[float, float]] = {}
    if isinstance(raw_positions, dict):
        for key, value in raw_positions.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            if key in graph:
                positions[key] = (float(value[0]), float(value[1]))
                continue
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            label = labels.get(idx)
            if label:
                positions[label] = (float(value[0]), float(value[1]))
    missing = set(graph.nodes()) - set(positions)
    if missing:
        seed = run_data.get('layout_seed')
        positions.update(
            nx.spring_layout(graph, seed=seed) if seed is not None else nx.spring_layout(graph)
        )
    return positions


def _matching_edges_from_paths(paths: Sequence[Sequence[str]]) -> Iterable[Tuple[str, str]]:
    for path in paths:
        if len(path) < 2:
            continue
        for u, v in zip(path, path[1:]):
            yield tuple(sorted((u, v)))


def visualize_matching_run(
    run_data: Dict[str, object],
    output_dir: str,
    *,
    figure_format: str = 'png',
    dpi: int = 150,
    skip_existing: bool = True,
) -> List[str]:
    """Render evolution frames for a single run.

    Parameters
    ----------
    run_data:
        Dictionary produced by :func:`matching_core.export_matching_run` or an
        equivalent data structure.
    output_dir:
        Directory where stage images will be saved.
    figure_format:
        Image format extension (``png``, ``svg`` etc.).
    dpi:
        Resolution for raster formats.
    skip_existing:
        If true, existing files will not be overwritten.

    Returns
    -------
    list of str
        Paths of all generated (or existing) image files.
    """

    os.makedirs(output_dir, exist_ok=True)

    labels = _extract_label_mapping(run_data)
    graph = _build_graph(run_data, labels)
    positions = _resolve_positions(graph, run_data, labels)

    matching_history = run_data.get('matching_history', []) or []
    tight_edges_history = run_data.get('tight_edges_history', []) or []
    blossom_history = run_data.get('blossom_history', []) or []

    stage_count = max(len(matching_history), len(tight_edges_history))

    color_map_raw = run_data.get('node_colors', {}) or {}
    default_color = run_data.get('default_node_color', 'lightblue')
    blossom_color = run_data.get('blossom_node_color', 'purple')

    generated_files: List[str] = []

    for stage_idx in range(stage_count):
        stage_file = os.path.join(output_dir, f'stage_{stage_idx:03d}.{figure_format}')
        if skip_existing and os.path.exists(stage_file):
            generated_files.append(stage_file)
            continue

        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_axis_off()

        nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=[color_map_raw.get(node, default_color) for node in graph.nodes()],
            node_size=run_data.get('node_size', 300),
            edgecolors='black',
            linewidths=0.5,
        )

        stage_data = matching_history[stage_idx] if stage_idx < len(matching_history) else {}
        stage_paths = stage_data.get('paths', []) if isinstance(stage_data, dict) else []
        matching_edges = {tuple(edge) for edge in _matching_edges_from_paths(stage_paths)}

        tight_edges_raw = (
            tight_edges_history[stage_idx] if stage_idx < len(tight_edges_history) else []
        )
        tight_edges = {tuple(sorted(tuple(edge))) for edge in tight_edges_raw}

        all_edges = {tuple(sorted(edge)) for edge in graph.edges()}
        tight_non_matching = tight_edges - matching_edges
        non_tight_edges = all_edges - tight_edges

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=list(non_tight_edges),
            edge_color='lightgray',
            width=1,
            alpha=0.4,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=list(tight_non_matching),
            edge_color='blue',
            width=2,
            style='dotted',
            alpha=0.8,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=list(matching_edges),
            edge_color='red',
            width=2.5,
        )

        blossom_nodes = set()
        for blossom in blossom_history:
            if blossom.get('stage', 0) <= stage_idx:
                for leaf in blossom.get('leaves', []) or []:
                    label = _label_for(int(leaf), labels)
                    blossom_nodes.add(label)
        if blossom_nodes:
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=list(blossom_nodes),
                node_color=blossom_color,
                node_size=run_data.get('blossom_node_size', 350),
                alpha=0.6,
            )

        nx.draw_networkx_labels(
            graph,
            positions,
            font_size=run_data.get('label_font_size', 10),
        )

        legend_elements = [
            Line2D([0], [0], color='red', lw=2.5, label='Matching edges'),
            Line2D([0], [0], color='blue', lw=1.5, ls='dotted', label='Tight edges'),
            Line2D([0], [0], color='gray', lw=1, alpha=0.4, label='Other edges'),
        ]
        if blossom_nodes:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', label='Blossom nodes',
                       markerfacecolor=blossom_color, markersize=8, alpha=0.6)
            )
        ax.legend(handles=legend_elements, loc='best')
        ax.set_title(f"Stage {stage_idx}")

        plt.tight_layout()
        plt.savefig(stage_file, dpi=dpi)
        plt.close()
        generated_files.append(stage_file)

    return generated_files


def visualize_runs_from_file(
    json_path: str,
    output_root: str = 'matching_visualizations',
    *,
    figure_format: str = 'png',
    dpi: int = 150,
    skip_existing: bool = True,
) -> Dict[str, List[str]]:
    """Render all runs contained in a JSON file."""

    with open(json_path, 'r') as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        data = [data]

    os.makedirs(output_root, exist_ok=True)
    outputs: Dict[str, List[str]] = {}

    for idx, run_data in enumerate(data):
        run_id = str(run_data.get('run_id', idx))
        run_dir = os.path.join(output_root, f'run_{run_id}')
        files = visualize_matching_run(
            run_data,
            run_dir,
            figure_format=figure_format,
            dpi=dpi,
            skip_existing=skip_existing,
        )
        outputs[run_id] = files
    return outputs


__all__ = [
    "visualize_matching_run",
    "visualize_runs_from_file",
]
