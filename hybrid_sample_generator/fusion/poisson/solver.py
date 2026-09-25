"""Sparse n-dimensional Poisson solver used by the fusion backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg


@dataclass(frozen=True, slots=True)
class SolverMetrics:
    """Diagnostics for one multi-channel Poisson solve."""

    unknowns: int
    iterations: tuple[int, ...]
    anchored_components: int


def _neighbor_offsets(ndim: int) -> tuple[tuple[int, ...], ...]:
    offsets = []
    for axis in range(ndim):
        for direction in (-1, 1):
            offset = [0] * ndim
            offset[axis] = direction
            offsets.append(tuple(offset))
    return tuple(offsets)


def _inside(coordinate: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    return all(0 <= value < shape[axis] for axis, value in enumerate(coordinate))


def solve_poisson(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    guidance_mode: str = "source",
    rtol: float = 1e-5,
    atol: float = 0.0,
    max_iterations: int = 2000,
) -> tuple[np.ndarray, SolverMetrics]:
    """Blend ``source`` into ``target`` inside an arbitrary 2D or 3D mask."""
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if source.shape != target.shape:
        raise ValueError(
            f"source shape {source.shape} does not match target shape {target.shape}."
        )
    if source.ndim not in (3, 4):
        raise ValueError(
            f"Poisson inputs must be (C,H,W) or (C,D,H,W), got {source.shape}."
        )
    if mask.shape != source.shape[1:]:
        raise ValueError(
            f"mask shape {mask.shape} does not match spatial shape {source.shape[1:]}."
        )
    if guidance_mode not in {"source", "mixed"}:
        raise ValueError("guidance_mode must be 'source' or 'mixed'.")
    if rtol < 0 or atol < 0 or (rtol == 0 and atol == 0):
        raise ValueError("rtol and atol must be non-negative and not both zero.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if not np.all(np.isfinite(source[:, mask])):
        raise ValueError("source contains non-finite values inside the Poisson mask.")
    if not np.all(np.isfinite(target)):
        raise ValueError("target contains non-finite values.")

    result = target.copy()
    unknowns = int(np.count_nonzero(mask))
    if unknowns == 0:
        return result, SolverMetrics(0, tuple(), 0)

    spatial_shape = tuple(int(value) for value in mask.shape)
    coordinates = [tuple(int(v) for v in point) for point in np.argwhere(mask)]
    index_map = np.full(spatial_shape, -1, dtype=np.int64)
    for index, coordinate in enumerate(coordinates):
        index_map[coordinate] = index

    channels = int(source.shape[0])
    rhs = np.zeros((channels, unknowns), dtype=np.float64)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    offsets = _neighbor_offsets(mask.ndim)

    structure = ndi.generate_binary_structure(mask.ndim, 1)
    component_labels, component_count = ndi.label(mask, structure=structure)
    has_boundary = np.zeros(component_count + 1, dtype=bool)

    for row, coordinate in enumerate(coordinates):
        degree = 0
        component = int(component_labels[coordinate])
        for offset in offsets:
            neighbor = tuple(
                coordinate[axis] + offset[axis] for axis in range(mask.ndim)
            )
            if not _inside(neighbor, spatial_shape):
                continue
            degree += 1

            source_gradient = source[(slice(None), *coordinate)] - source[
                (slice(None), *neighbor)
            ]
            if guidance_mode == "mixed":
                target_gradient = target[(slice(None), *coordinate)] - target[
                    (slice(None), *neighbor)
                ]
                guidance = np.where(
                    np.abs(source_gradient) >= np.abs(target_gradient),
                    source_gradient,
                    target_gradient,
                )
            else:
                guidance = source_gradient
            rhs[:, row] += guidance

            neighbor_index = int(index_map[neighbor])
            if neighbor_index >= 0:
                rows.append(row)
                columns.append(neighbor_index)
                values.append(-1.0)
            else:
                has_boundary[component] = True
                rhs[:, row] += target[(slice(None), *neighbor)]

        rows.append(row)
        columns.append(row)
        values.append(float(degree))

    anchored_components = 0
    for component in range(1, component_count + 1):
        if has_boundary[component]:
            continue
        anchor_coordinate = tuple(
            int(value) for value in np.argwhere(component_labels == component)[0]
        )
        anchor_index = int(index_map[anchor_coordinate])
        rows.append(anchor_index)
        columns.append(anchor_index)
        values.append(1.0)
        rhs[:, anchor_index] += target[(slice(None), *anchor_coordinate)]
        anchored_components += 1

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(unknowns, unknowns),
        dtype=np.float64,
    ).tocsr()

    iteration_counts = []
    for channel in range(channels):
        iterations = 0

        def count_iteration(_):
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            matrix,
            rhs[channel],
            x0=target[channel][mask].astype(np.float64, copy=False),
            rtol=float(rtol),
            atol=float(atol),
            maxiter=int(max_iterations),
            callback=count_iteration,
        )
        if info != 0:
            reason = (
                f"did not converge after {info} iterations"
                if info > 0
                else f"failed with solver status {info}"
            )
            raise RuntimeError(
                f"Poisson conjugate-gradient solve for channel {channel} {reason}."
            )
        result[channel][mask] = solution.astype(np.float32, copy=False)
        iteration_counts.append(iterations)

    return result, SolverMetrics(
        unknowns=unknowns,
        iterations=tuple(iteration_counts),
        anchored_components=anchored_components,
    )


__all__ = ["SolverMetrics", "solve_poisson"]
