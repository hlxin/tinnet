"""Adsorbate specifications for the Newns-Anderson chemisorption module.

An :class:`AdsorbateSpec` is *pure data*.  It tells the generic
:class:`~tinnet.tinnet.physics.newns_anderson.NewnsAndersonModel` which
adsorbate frontier orbitals couple to the metal d band and with what
degeneracy, and it tells the :class:`~tinnet.tinnet.adsorption_energy.AdsorptionEnergy`
facade how to build the adsorbed structure and which pretrained ensemble to
load.  Supporting a new adsorbate therefore means registering a spec, not
editing the theory code::

    from tinnet.tinnet.physics import OrbitalSpec, AdsorbateSpec, register_adsorbate

    register_adsorbate(AdsorbateSpec(
        name='N',
        orbitals=(OrbitalSpec('pz', r'p_{z}', degeneracy=1, alpha=0.07),
                  OrbitalSpec('pxy', r'p_{xy}', degeneracy=2, alpha=0.05)),
        esp=-4.0,
        geometry=(('N', (0.0, 0.0, 1.5)),),
        d_band='tabulated',
        gnn=dict(atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1),
        checkpoint=('adsorption_energy', 'N', 'atop'),
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator, Mapping


@dataclass(frozen=True)
class OrbitalSpec:
    """One adsorbate frontier orbital (or degenerate set) in the Newns-Anderson model.

    Parameters
    ----------
    name
        Short identifier, e.g. ``'3sigma'`` or ``'pz'``.
    label
        LaTeX subscript used in plots, e.g. ``r'3\\sigma'``.
    degeneracy
        Number of degenerate orbitals sharing the same parameters (``2`` for a
        :math:`\\pi` or :math:`p_{xy}` pair).
    alpha
        Orthogonalization (Pauli repulsion) coefficient multiplying
        :math:`2 (n_a + f)\\,\\beta\\,V_{ad}^2`.
    """

    name: str
    label: str
    degeneracy: int = 1
    alpha: float = 0.0

    @property
    def parameter_names(self) -> tuple[str, str, str]:
        return (f"adse_{self.name}", f"beta_{self.name}", f"delta_{self.name}")

    @property
    def parameter_labels(self) -> tuple[str, str, str]:
        return (
            rf"$\epsilon_{{{self.label}}}$",
            rf"$\beta_{{{self.label}}}$",
            rf"$\delta_{{{self.label}}}$",
        )


@dataclass(frozen=True)
class AdsorbateSpec:
    """Everything needed to run the Newns-Anderson TinNet for one adsorbate/site.

    Theory fields (consumed by ``NewnsAndersonModel``)
    --------------------------------------------------
    orbitals
        Frontier orbitals coupled to the d band, in latent-output order.
    esp
        Constant sp-band / reference contribution added to the energy (eV).
    d_band
        ``'latent'``: the GNN predicts the d-band center and half width.
        ``'tabulated'``: they are supplied as constants (for example from the
        :class:`BandCenter` model), and the GNN only predicts orbital terms.

    Structure fields (consumed by the ``AdsorptionEnergy`` facade)
    ---------------------------------------------------------------
    geometry
        ``((symbol, (dx, dy, dz)), ...)`` offsets of adsorbate atoms relative
        to the adsorption site.
    site
        Site type label (currently ``'atop'``).

    Model fields
    ------------
    gnn
        Hyperparameters of the pretrained CGCNN ensemble.
    checkpoint
        Path parts below ``data/pretrained`` for the ensemble.
    n_ensemble
        Number of ensemble members.
    """

    name: str
    orbitals: tuple[OrbitalSpec, ...]
    esp: float
    geometry: tuple[tuple[str, tuple[float, float, float]], ...]
    site: str = "atop"
    d_band: str = "latent"
    gnn: Mapping[str, int] = field(default_factory=dict)
    checkpoint: tuple[str, ...] = ()
    n_ensemble: int = 10
    energy_label: str | None = None

    def __post_init__(self):
        if self.d_band not in ("latent", "tabulated"):
            raise ValueError("d_band must be 'latent' or 'tabulated'.")
        if not self.orbitals:
            raise ValueError("An adsorbate needs at least one frontier orbital.")
        object.__setattr__(self, "orbitals", tuple(self.orbitals))
        object.__setattr__(self, "checkpoint", tuple(self.checkpoint))
        object.__setattr__(self, "gnn", dict(self.gnn))

    # -- derived quantities -------------------------------------------------
    @property
    def phys_model(self) -> str:
        """Legacy identifier such as ``'OH_atop'``."""
        return f"{self.name}_{self.site}"

    @property
    def elements(self) -> frozenset[str]:
        """Chemical symbols that belong to the adsorbate."""
        return frozenset(symbol for symbol, _ in self.geometry)

    @property
    def n_orbitals(self) -> int:
        return len(self.orbitals)

    @property
    def n_latent(self) -> int:
        """Raw GNN outputs: 3 per orbital (+2 if the d band is learned)."""
        return 3 * self.n_orbitals + (2 if self.d_band == "latent" else 0)

    @property
    def constant_names(self) -> tuple[str, ...]:
        if self.d_band == "tabulated":
            return ("vad2", "d_cen", "width")
        return ("vad2",)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = ["vad2", "d_cen", "width"]
        for orbital in self.orbitals:
            names.extend(orbital.parameter_names)
        return tuple(names)

    @property
    def parameter_labels(self) -> tuple[str, ...]:
        labels = [r"$V_{ad}^{2}$", r"$\epsilon_{d}$", r"$W_{d}$"]
        for orbital in self.orbitals:
            labels.extend(orbital.parameter_labels)
        return tuple(labels)

    @property
    def property_label(self) -> str:
        if self.energy_label is not None:
            return self.energy_label
        return rf"$E_{{ad}}^{{{self.name}, {self.site}}}$ (eV)"

    def with_changes(self, **changes) -> "AdsorbateSpec":
        """Return a copy with some fields replaced."""
        return replace(self, **changes)


# ----------------------------------------------------------------------
# Built-in adsorbates (parameters of the published TinNet checkpoints)
# ----------------------------------------------------------------------
_ALPHA_OH = 0.06378761202273762

OH_ATOP = AdsorbateSpec(
    name="OH",
    orbitals=(
        OrbitalSpec("3sigma", r"3\sigma", degeneracy=1, alpha=_ALPHA_OH),
        OrbitalSpec("1pi", r"1\pi", degeneracy=2, alpha=_ALPHA_OH),
        OrbitalSpec("4sigma*", r"4\sigma^{*}", degeneracy=1, alpha=_ALPHA_OH),
    ),
    esp=-2.693696878597913,
    geometry=(("O", (0.0, 0.0, 2.00)), ("H", (0.8, 0.0, 2.41))),
    d_band="latent",
    gnn=dict(atom_fea_len=95, n_conv=5, h_fea_len=174, n_h=2),
    checkpoint=("adsorption_energy", "OH", "atop"),
    energy_label=r"$E_{ad}^{OH, top}$ (eV)",
)

O_ATOP = AdsorbateSpec(
    name="O",
    orbitals=(
        OrbitalSpec("pz", r"pz", degeneracy=1, alpha=0.07889181751783157),
        OrbitalSpec("pxy", r"pxy", degeneracy=2, alpha=0.05687347683456299),
    ),
    esp=-3.765294246337454,
    geometry=(("O", (0.0, 0.0, 1.80)),),
    d_band="tabulated",
    gnn=dict(atom_fea_len=71, n_conv=7, h_fea_len=104, n_h=4),
    checkpoint=("adsorption_energy", "O", "atop"),
    energy_label=r"$E_{ad}^{O, top}$ (eV)",
)

ADSORBATE_REGISTRY: dict[str, AdsorbateSpec] = {}


def register_adsorbate(spec: AdsorbateSpec, *, overwrite: bool = False) -> AdsorbateSpec:
    """Register an adsorbate spec under ``spec.name`` (and ``spec.phys_model``)."""
    if spec.name in ADSORBATE_REGISTRY and not overwrite:
        raise ValueError(
            f"Adsorbate '{spec.name}' is already registered; pass overwrite=True to replace it."
        )
    ADSORBATE_REGISTRY[spec.name] = spec
    return spec


def adsorbate_spec(name: str | None = None, *, phys_model: str | None = None) -> AdsorbateSpec:
    """Look up a spec by adsorbate name (``'OH'``) or legacy model name (``'OH_atop'``)."""
    if name is None and phys_model is None:
        raise ValueError("Provide an adsorbate name or a phys_model identifier.")
    if name is not None:
        if name in ADSORBATE_REGISTRY:
            return ADSORBATE_REGISTRY[name]
        raise ValueError(
            f"Unsupported adsorbate '{name}'. Supported adsorbates: {', '.join(available_adsorbates())}."
        )
    for spec in ADSORBATE_REGISTRY.values():
        if spec.phys_model == phys_model:
            return spec
    raise ValueError(f"Unsupported physical model: {phys_model}")


def available_adsorbates() -> list[str]:
    return sorted(ADSORBATE_REGISTRY)


def iter_adsorbates() -> Iterator[AdsorbateSpec]:
    return iter(ADSORBATE_REGISTRY.values())


register_adsorbate(OH_ATOP)
register_adsorbate(O_ATOP)
