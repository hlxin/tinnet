"""Shared utilities for the TinNet model scripts.

The functions in this module intentionally avoid changing model architecture,
checkpoint names, and physics equations.  They only centralize recurring
engineering tasks such as path resolution, device handling, ASE-Atoms copying,
and publication-style plotting defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams

ADSORBATE_SYMBOLS = {"O", "H"}


def project_root() -> Path:
    """Return the package/project root that should contain the ``data`` folder.

    The original TinNet scripts used paths such as ``./data/pretrained/...``.
    That works only when the notebook is executed from the package root.  This
    resolver keeps the old behavior but also supports importing the package from
    another working directory.
    """
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for base in candidates:
        if (base / "data").exists():
            return base
    # Fall back to the current working directory so missing-file errors still
    # mention the location users are likely running from.
    return Path.cwd()


def data_path(*parts: str) -> Path:
    """Resolve a path below the TinNet ``data`` directory."""
    return project_root() / "data" / Path(*parts)


def pretrained_path(model_family: str, *parts: str) -> Path:
    """Resolve a path below ``data/pretrained/<model_family>``."""
    return data_path("pretrained", model_family, *parts)


def torch_device() -> torch.device:
    """Return the active PyTorch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(tensor: torch.Tensor, device: torch.device | None = None, non_blocking: bool = True) -> torch.Tensor:
    """Move a tensor to the selected device without changing its dtype."""
    if device is None:
        device = torch_device()
    return tensor.to(device=device, non_blocking=non_blocking)


def as_float_tensor(values, device: torch.device | None = None) -> torch.Tensor:
    """Create a float32 tensor on the requested device."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return to_device(tensor, device) if device is not None else tensor


def as_long_tensor(values, device: torch.device | None = None) -> torch.Tensor:
    """Create an int64 tensor on the requested device."""
    tensor = torch.as_tensor(values, dtype=torch.long)
    return to_device(tensor, device) if device is not None else tensor


def load_checkpoint_state(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device | None = None) -> dict:
    """Load a checkpoint into ``model`` and return the checkpoint dictionary.

    Parameters
    ----------
    model
        Model instance whose ``state_dict`` should be populated.
    checkpoint_path
        Path to a TinNet checkpoint file.
    device
        Target device.  CPU loading uses ``map_location`` explicitly so that
        CUDA-trained checkpoints can be used on machines without a GPU.
    """
    if device is None:
        device = torch_device()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def copy_without_adsorbates(image, symbols: Iterable[str] = ADSORBATE_SYMBOLS):
    """Return a copy of an ASE ``Atoms`` object with selected adsorbates removed.

    The original scripts deleted O/H atoms in-place.  Returning a copy prevents
    repeated predictions from becoming order-dependent.
    """
    if image is None:
        raise ValueError("image must be provided.")
    symbols = set(symbols)
    clean_image = image.copy()
    adsorbate_indices = [idx for idx, atom in enumerate(clean_image) if atom.symbol in symbols]
    for idx in sorted(adsorbate_indices, reverse=True):
        del clean_image[idx]
    return clean_image


def normalize_atom_index(atom_inx: int | np.integer) -> int:
    """Validate and normalize a single atom index."""
    if atom_inx is None:
        raise ValueError("atom_inx must be provided.")
    if not isinstance(atom_inx, (int, np.integer)):
        raise TypeError(f"atom_inx must be an integer, got {type(atom_inx).__name__}.")
    atom_inx = int(atom_inx)
    if atom_inx < 0:
        raise IndexError("atom_inx must be non-negative.")
    return atom_inx


def normalize_site_indices(site_inx: Sequence[int] | np.ndarray | int) -> list[int]:
    """Normalize a site-index input into a list of Python integers."""
    if site_inx is None:
        raise ValueError("site_inx must be provided.")
    if isinstance(site_inx, np.ndarray):
        site_inx = site_inx.tolist()
    elif isinstance(site_inx, (int, np.integer)):
        site_inx = [int(site_inx)]
    else:
        site_inx = list(site_inx)
    site_inx = [int(idx) for idx in site_inx]
    if any(idx < 0 for idx in site_inx):
        raise IndexError("site indices must be non-negative.")
    return site_inx


def validate_atom_index(image, atom_inx: int, label: str = "atom_inx") -> None:
    """Raise a clear error if an atom/site index is outside an ASE image."""
    if atom_inx >= len(image):
        raise IndexError(f"{label}={atom_inx} is outside the structure with {len(image)} atoms.")



def stack_scalar_parameters(values, like: torch.Tensor | None = None) -> torch.Tensor:
    """Stack scalar parameters while safely detaching autograd tensors.

    This helper is used for values returned as interpretable model parameters.
    They are outputs for analysis, not tensors that should keep an autograd graph.
    """
    if like is not None:
        dtype = like.dtype
        device = like.device
    else:
        dtype = torch.float32
        device = torch_device()
    return torch.stack([
        torch.as_tensor(value, dtype=dtype, device=device).detach().reshape(())
        for value in values
    ])

def values_for_symbols(symbols: Sequence[str], values: Mapping[str, float], name: str) -> list[float]:
    """Resolve per-element constants and fail with a clear message if missing."""
    missing = sorted({symbol for symbol in symbols if symbol not in values})
    if missing:
        raise KeyError(f"Missing {name} constants for element(s): {', '.join(missing)}")
    return [values[symbol] for symbol in symbols]


def set_publication_style() -> None:
    """Apply the Matplotlib style used by the original TinNet figures."""
    rcParams["ps.useafm"] = True
    plt.rc("font", **{"family": "sans-serif", "sans-serif": ["DejaVu Sans"]})
    rcParams["pdf.fonttype"] = 42
    rcParams["errorbar.capsize"] = 4.0
    mpl.rcParams["ytick.major.width"] = 0.5
    mpl.rcParams["ytick.minor.width"] = 0.5
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["xtick.minor.width"] = 0.5
    matplotlib.rc("xtick.major", size=4)
    matplotlib.rc("xtick.minor", size=2)
    matplotlib.rc("ytick.major", size=4)
    matplotlib.rc("ytick.minor", size=2)
    matplotlib.rc("lines", linewidth=0.5)
    matplotlib.rc("lines", markeredgewidth=0.5)
    matplotlib.rc("font", size=7)
    plt.rcParams["axes.linewidth"] = 0.5




def make_data_loader(dataset, collate_fn=None, batch_size=256, sampler=None, num_workers=0, pin_memory=False):
    """Create a DataLoader with the default TinNet sampler convention.

    A one-line wrapper keeps small inference-only loaders consistent across the
    model scripts without duplicating boilerplate.
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.dataloader import default_collate
    from torch.utils.data.sampler import SubsetRandomSampler

    if collate_fn is None:
        collate_fn = default_collate
    if sampler is None:
        sampler = SubsetRandomSampler(np.arange(len(dataset)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def move_tensor_sequence(values, device: torch.device, non_blocking: bool = True):
    """Move a sequence of tensors to a device while preserving list structure."""
    return [value.to(device, non_blocking=non_blocking) for value in values]


def stack_dense_by_ids(tensors, ids, device: torch.device | None = None) -> torch.Tensor:
    """Stack sparse/dense tensors selected by integer ids and move to ``device``."""
    stacked = torch.stack([tensors[int(idx)].to_dense() for idx in ids])
    return stacked.to(device) if device is not None else stacked


def sign_arrays(values, positive: str = "blue", negative: str = "red"):
    """Return color and signed-label arrays for a vector of values."""
    values = np.asarray(values, dtype=float)
    colors = [signed_color(value, positive=positive, negative=negative) for value in values]
    labels = [signed_label(value) for value in values]
    return colors, labels


def draw_horizontal_waterfall(
    ax,
    *,
    baseline: float,
    deltas,
    labels,
    xlabel: str,
    ref_name: str = "Reference",
    target_name: str = "Target",
    left_annotations=None,
    right_annotations=None,
    y_positions=None,
    connector_top_pad: float = 1.5,
    arrow_width: float = 1.0 / 3.0,
):
    """Draw the horizontal waterfall style used in TinNet SHAP figures.

    Parameters are intentionally primitive so adsorption, band-center, and
    cohesive-energy explanations can share the same plotting primitive without
    coupling to each model's internal data structures.
    """
    deltas = np.asarray(deltas, dtype=float)
    if y_positions is None:
        y_positions = np.arange(len(deltas), 0, -1)
    else:
        y_positions = np.asarray(y_positions, dtype=float)

    starts = float(baseline) + np.concatenate(([0.0], np.cumsum(deltas[:-1])))
    final_value = starts[-1] + deltas[-1] if len(deltas) else float(baseline)

    for start, delta, y in zip(starts, deltas, y_positions):
        color = signed_color(delta)
        ax.arrow(
            x=start,
            y=y,
            dx=delta,
            dy=0,
            color=color,
            width=arrow_width,
            head_width=arrow_width,
            head_length=0.15 * abs(delta),
            length_includes_head=True,
        )

    def _annotate(items, x):
        if not items:
            return
        for value, y in items:
            ax.annotate(
                signed_label(value),
                xy=(x, y),
                xycoords=("axes fraction", "data"),
                ha="center",
                va="center",
                color=signed_color(value),
            )

    if left_annotations is None:
        left_annotations = list(zip(deltas, y_positions))
    if right_annotations is not None:
        _annotate(right_annotations, 1.12)
    _annotate(left_annotations, -0.12)

    ax.set_ylim([0.5, len(deltas) + 0.5])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params("y", length=0, width=0, which="major")

    top = len(deltas) + connector_top_pad
    ax.plot([baseline, baseline], [top, 0.5], "--", color="gray", linewidth=1)
    for i, x in enumerate(starts[1:], start=1):
        y_upper = y_positions[i - 1] - arrow_width
        y_lower = y_positions[i] + arrow_width
        ax.plot([x, x], [y_lower, y_upper], "--", color="gray", linewidth=1)
    ax.plot([final_value, final_value], [top, 0.5], "--", color="orange", linewidth=1)

    ax.annotate(f"{ref_name}\n{baseline:.2f}", xy=(baseline, 1.15),
                xycoords=("data", "axes fraction"), ha="center", va="center", color="gray")
    ax.annotate(f"{target_name}\n{final_value:.2f}", xy=(final_value, 1.05),
                xycoords=("data", "axes fraction"), ha="center", va="center", color="orange")
    return final_value


# -----------------------------------------------------------------------------
# Shared element-property tables
# -----------------------------------------------------------------------------

# Solid-state-table properties used by the adsorption-energy TinNet models.
# ``vad2`` follows the LMTO-derived V_ad^2 scale used in the original scripts.
ADSORPTION_ATOM_PROPERTIES = {
    "Ca": {"f": 0.1, "vad2": 20.8, "eps_d": None},
    "Sc": {"f": 0.2, "vad2": 7.90, "eps_d": None},
    "Ti": {"f": 0.3, "vad2": 4.65, "eps_d": 1.50},
    "V": {"f": 0.4, "vad2": 3.15, "eps_d": 1.06},
    "Cr": {"f": 0.5, "vad2": 2.35, "eps_d": 0.16},
    "Mn": {"f": 0.6, "vad2": 1.94, "eps_d": 0.07},
    "Fe": {"f": 0.7, "vad2": 1.59, "eps_d": -0.92},
    "Co": {"f": 0.8, "vad2": 1.34, "eps_d": -1.17},
    "Ni": {"f": 0.9, "vad2": 1.16, "eps_d": -1.29},
    "Cu": {"f": 1.0, "vad2": 1.00, "eps_d": -2.67},
    "Zn": {"f": 1.0, "vad2": 0.46, "eps_d": None},
    "Sr": {"f": 0.1, "vad2": 36.5, "eps_d": None},
    "Y": {"f": 0.2, "vad2": 17.3, "eps_d": None},
    "Zr": {"f": 0.3, "vad2": 10.90, "eps_d": 1.95},
    "Nb": {"f": 0.4, "vad2": 7.73, "eps_d": 1.41},
    "Mo": {"f": 0.5, "vad2": 6.62, "eps_d": 0.35},
    "Tc": {"f": 0.6, "vad2": 4.71, "eps_d": -0.60},
    "Ru": {"f": 0.7, "vad2": 3.87, "eps_d": -1.41},
    "Rh": {"f": 0.8, "vad2": 3.32, "eps_d": -1.73},
    "Pd": {"f": 0.9, "vad2": 2.78, "eps_d": -1.83},
    "Ag": {"f": 1.0, "vad2": 2.26, "eps_d": -4.30},
    "Cd": {"f": 1.0, "vad2": 1.58, "eps_d": None},
    "Ba": {"f": 0.1, "vad2": 41.5, "eps_d": None},
    "Lu": {"f": 0.2, "vad2": 17.1, "eps_d": None},
    "Hf": {"f": 0.3, "vad2": 11.90, "eps_d": 2.47},
    "Ta": {"f": 0.4, "vad2": 9.05, "eps_d": 2.00},
    "W": {"f": 0.5, "vad2": 7.27, "eps_d": 0.77},
    "Re": {"f": 0.6, "vad2": 6.04, "eps_d": -0.51},
    "Os": {"f": 0.7, "vad2": 5.13, "eps_d": None},
    "Ir": {"f": 0.8, "vad2": 4.45, "eps_d": -2.11},
    "Pt": {"f": 0.9, "vad2": 3.90, "eps_d": -2.25},
    "Au": {"f": 1.0, "vad2": 3.35, "eps_d": -3.56},
    "Hg": {"f": 1.0, "vad2": 2.64, "eps_d": None},
}

# Reference electronic-structure properties used by band-center and band-moment models.
REFERENCE_MATERIAL_PROPERTIES = {
    "Ag": {"d_cen": -4.1421, "full_width": 4.5040, "rd": 0.6606606606606606, "bulk_filling": 9.755790629075912213e-01},
    "Au": {"d_cen": -3.6577, "full_width": 5.8312, "rd": 0.7607607607607607, "bulk_filling": 9.638752405788268973e-01},
    "Co": {"d_cen": -1.5905, "full_width": 6.6969, "rd": 0.5605605605605606, "bulk_filling": 7.711781551994808526e-01},
    "Cr": {"d_cen": -0.0876, "full_width": 7.2096, "rd": 0.6306306306306306, "bulk_filling": 5.026890307897071697e-01},
    "Cu": {"d_cen": -2.6521, "full_width": 4.2446, "rd": 0.4904904904904905, "bulk_filling": 9.687762969001947333e-01},
    "Fe": {"d_cen": -0.9278, "full_width": 7.1120, "rd": 0.6206206206206206, "bulk_filling": 6.842646244058228078e-01},
    "Ir": {"d_cen": -2.6636, "full_width": 9.5022, "rd": 0.8208208208208209, "bulk_filling": 7.713775689575533834e-01},
    "Mn": {"d_cen": -0.6036, "full_width": 7.0846, "rd": 0.5905905905905906, "bulk_filling": 5.625623395638867930e-01},
    "Mo": {"d_cen": -0.0110, "full_width": 9.0573, "rd": 0.8608608608608609, "bulk_filling": 5.085650613858612168e-01},
    "Nb": {"d_cen": 0.6878, "full_width": 8.9762, "rd": 0.9409409409409409, "bulk_filling": 4.053595224934098407e-01},
    "Ni": {"d_cen": -1.6686, "full_width": 5.3541, "rd": 0.5205205205205206, "bulk_filling": 8.642030365513106993e-01},
    "Os": {"d_cen": -1.9693, "full_width": 10.6399, "rd": 0.8508508508508509, "bulk_filling": 6.868617591064947181e-01},
    "Pd": {"d_cen": -2.0870, "full_width": 5.6793, "rd": 0.6706706706706707, "bulk_filling": 9.147420170513048676e-01},
    "Pt": {"d_cen": -2.6369, "full_width": 7.6381, "rd": 0.7907907907907907, "bulk_filling": 8.750916806647935919e-01},
    "Re": {"d_cen": -1.0633, "full_width": 10.9953, "rd": 0.8808808808808809, "bulk_filling": 5.905781396979362663e-01},
    "Rh": {"d_cen": -2.0874, "full_width": 7.3926, "rd": 0.7207207207207207, "bulk_filling": 7.951773878664225581e-01},
    "Ru": {"d_cen": -1.6305, "full_width": 8.0663, "rd": 0.7507507507507507, "bulk_filling": 7.060335740069819677e-01},
    "Sc": {"d_cen": 1.7032, "full_width": 5.9997, "rd": 0.9409409409409409, "bulk_filling": 1.936272929841076074e-01},
    "Ta": {"d_cen": 1.1906, "full_width": 10.8913, "rd": 1.021021021021021, "bulk_filling": 3.786712857939578680e-01},
    "Ti": {"d_cen": 1.3243, "full_width": 6.7421, "rd": 0.7907907907907907, "bulk_filling": 2.653109539307398901e-01},
    "V": {"d_cen": 0.5548, "full_width": 6.9774, "rd": 0.6906906906906907, "bulk_filling": 3.818390980189110273e-01},
    "W": {"d_cen": 0.1732, "full_width": 10.7373, "rd": 0.9409409409409409, "bulk_filling": 4.850581815594974255e-01},
    "Y": {"d_cen": 2.2707, "full_width": 7.8737, "rd": 1.2512512512512513, "bulk_filling": 1.875853315789731690e-01},
    "Zr": {"d_cen": 1.6485, "full_width": 8.9050, "rd": 1.0710710710710711, "bulk_filling": 2.920224652312821134e-01},
}


def adsorption_atom_properties() -> dict:
    """Return a copy of the adsorption-model element-property table."""
    return {element: props.copy() for element, props in ADSORPTION_ATOM_PROPERTIES.items()}


def material_properties() -> dict:
    """Return the universal material-property dictionary used by TinNet.

    The base data are loaded from ``data/MaterialDict.pkl`` because it contains
    atomic quantities such as ionization potentials and electron affinities.
    The reference d-band quantities used in the TinNet publications are then
    merged into that dictionary so band-center and band-moment models call the
    same source of truth.
    """
    import pickle

    with open(data_path("MaterialDict.pkl"), "rb") as file:
        raw = pickle.load(file, encoding="bytes")
    data = {key.decode("utf8") if isinstance(key, bytes) else key: value for key, value in raw.items()}
    for element, props in REFERENCE_MATERIAL_PROPERTIES.items():
        data.setdefault(element, {})
        data[element].update(props)
    return data

def atom_features() -> dict[int, list[int]]:
    """Return the default TinNet/CGCNN atomic feature vectors.

    The original TinNet scripts embedded this large ``atom_init``-style table
    independently in multiple modules.  Keeping it in one utility function
    prevents silent descriptor drift across adsorption-energy, d-band-center,
    d-band-moment, and cohesive-energy models.
    """
    atom_fea_dict = {
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0],
        3: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0],
        4: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0],
        5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0],
        6: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0],
        8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0],
        9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0],
        10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        11: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        12: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        13: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        14: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        15: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        16: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        17: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        18: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        19: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        20: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        21: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        22: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        23: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        24: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        25: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        26: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        27: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        28: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        29: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        30: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        31: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        32: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        33: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        34: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        35: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        36: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        37: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        38: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        39: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        40: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        41: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        42: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        43: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        44: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        45: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        46: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        47: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        48: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        49: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        50: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        51: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        52: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        53: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        54: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        55: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        56: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        57: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        58: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        59: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        60: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        61: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        62: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        63: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        64: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        65: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        66: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        67: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        68: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        69: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        70: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        71: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        72: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        73: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        74: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        75: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        76: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        77: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        78: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        79: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        80: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        81: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        82: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        83: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        84: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        85: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        86: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        87: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        88: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        89: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        90: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        91: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        92: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        93: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        94: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        95: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        96: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        97: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        98: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        99: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        100: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    return atom_fea_dict


# Cohesive-energy constants used by the renormalized-atom cohesion module.
# ``promotion_energy`` is the atomic promotion energy, and ``wigner_seitz_volume``
# is the per-element Wigner-Seitz volume used in the free-electron conduction-band term.
COHESIVE_ENERGY_CONSTANTS = {
    "Sc": {"wigner_seitz_volume": 24.724542268797304, "promotion_energy": 2.2603424},
    "Ti": {"wigner_seitz_volume": 17.424646217258488, "promotion_energy": 2.440124875909091},
    "V": {"wigner_seitz_volume": 13.917982774094817, "promotion_energy": 2.6619930170731707},
    "Cr": {"wigner_seitz_volume": 11.863283203031445, "promotion_energy": 3.6188933459259256},
    "Mn": {"wigner_seitz_volume": 10.785494076584158, "promotion_energy": 4.265695346153847},
    "Fe": {"wigner_seitz_volume": 11.451083083113335, "promotion_energy": 2.9840186344444444},
    "Co": {"wigner_seitz_volume": 10.398800712477017, "promotion_energy": 1.7877074285714285},
    "Ni": {"wigner_seitz_volume": 10.916490557546439, "promotion_energy": 0.19241192499999998},
    "Cu": {"wigner_seitz_volume": 12.14350679198855, "promotion_energy": 0.0},
    "Y": {"wigner_seitz_volume": 32.44397679442938, "promotion_energy": 1.9629624225},
    "Zr": {"wigner_seitz_volume": 23.170475005920792, "promotion_energy": 1.7412473684210525},
    "Nb": {"wigner_seitz_volume": 18.71258535775407, "promotion_energy": 1.544762289473684},
    "Mo": {"wigner_seitz_volume": 16.197202890320533, "promotion_energy": 2.9535343183673466},
    "Tc": {"wigner_seitz_volume": None, "promotion_energy": 2.330626157894737},
    "Ru": {"wigner_seitz_volume": 14.229567963011105, "promotion_energy": 1.4199372666666668},
    "Rh": {"wigner_seitz_volume": 14.229567963011105, "promotion_energy": 1.1374149333333332},
    "Pd": {"wigner_seitz_volume": 15.35668980826143, "promotion_energy": 1.03267305},
    "Ag": {"wigner_seitz_volume": 17.969127631160283, "promotion_energy": 0.0},
    "Hf": {"wigner_seitz_volume": None, "promotion_energy": 3.3832650593749998},
    "Ta": {"wigner_seitz_volume": 18.71258535775407, "promotion_energy": 2.5171466666666666},
    "W": {"wigner_seitz_volume": 16.54181279434536, "promotion_energy": 3.9899073974358963},
    "Re": {"wigner_seitz_volume": 14.70561133998898, "promotion_energy": 3.3005202272727274},
    "Os": {"wigner_seitz_volume": 14.229567963011105, "promotion_energy": 1.9706242272727275},
    "Ir": {"wigner_seitz_volume": 14.545769144994983, "promotion_energy": 1.8323957333333332},
    "Pt": {"wigner_seitz_volume": 15.857412626116277, "promotion_energy": 0.767994218},
    "Au": {"wigner_seitz_volume": 18.153110451663995, "promotion_energy": 0.0},
}


def cohesive_energy_constants() -> dict[str, dict[str, float]]:
    """Return a copy of constants used by the cohesive-energy theory module."""
    return {element: values.copy() for element, values in COHESIVE_ENERGY_CONSTANTS.items()}


def cohesive_constant_for_symbols(symbols: Sequence[str], key: str) -> list[float]:
    """Return per-symbol cohesive-energy constants for a given property key."""
    table = cohesive_energy_constants()
    missing = sorted({symbol for symbol in symbols if symbol not in table or table[symbol].get(key) is None})
    if missing:
        raise KeyError(f"Missing cohesive-energy constant '{key}' for element(s): {', '.join(missing)}")
    return [table[symbol][key] for symbol in symbols]


def signed_color(value: float, positive: str = "blue", negative: str = "red") -> str:
    """Return a plotting color by the sign convention used in TinNet SHAP figures."""
    return negative if value < 0 else positive


def signed_label(value: float, decimals: int = 2) -> str:
    """Format a signed magnitude label such as '+0.12' or '-0.12'."""
    sign = "-" if value < 0 else "+"
    return f"{sign}{abs(value):.{decimals}f}"


def save_figure(fig, plot_name: str, save_fig: str | None = "png") -> None:
    """Save a Matplotlib figure using the original TinNet export conventions."""
    if save_fig is None:
        return
    if save_fig == "png":
        fig.savefig(f"{plot_name}.png", bbox_inches="tight", dpi=600)
    elif save_fig == "pdf":
        fig.savefig(f"{plot_name}.pdf", bbox_inches="tight")
    else:
        raise ValueError("save_fig must be one of {'png', 'pdf', None}.")


def flatten_nested(values, dtype=torch.float32) -> torch.Tensor:
    """Flatten a list of per-structure constants into one tensor."""
    flat = [item for sublist in values for item in sublist]
    return torch.as_tensor(flat, dtype=dtype)
