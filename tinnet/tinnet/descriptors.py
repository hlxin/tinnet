"""Crystal-graph descriptors shared by TinNet models."""

from __future__ import annotations

import numpy as np
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from .tinnet_utils import atom_features
except ImportError:  # Allows running this file directly during debugging.
    from tinnet_utils import atom_features


class VoronoiGraph:
    """Voronoi-connectivity crystal graph with Gaussian-expanded bond weights.

    Parameters
    ----------
    max_num_nbr : int
        Maximum number of neighbors per atom in the graph.
    radius, dmin, step : float
        Gaussian basis for the bond feature (``np.arange(dmin, radius + step, step)``).
    dict_atom_fea : dict or None
        Atomic-number -> feature-vector table (defaults to the CGCNN ``atom_init`` table).

    ``feas(image)`` returns ``(atom_fea (n, F), nbr_fea (n, M, K), nbr_fea_idx (n, M))``.
    """

    def __init__(self, max_num_nbr=12, radius=8.0, dmin=0.0, step=0.2, dict_atom_fea=None):
        self.max_num_nbr = max_num_nbr
        self.step = step
        self.dict_atom_fea = atom_features() if dict_atom_fea is None else dict_atom_fea
        assert dmin < radius
        assert radius - dmin > self.step
        self.filter = np.arange(dmin, radius + self.step, self.step)

    def feas(self, image):
        try:
            image = AseAtomsAdaptor.get_structure(image)
        except Exception:
            pass

        atom_fea = np.array([self.dict_atom_fea[i] for i in image.atomic_numbers])

        conn = VoronoiConnectivity(image).connectivity_array
        all_nbrs = []
        for ii in range(0, conn.shape[0]):
            curnbr = []
            for jj in range(0, conn.shape[1]):
                for kk in range(0, conn.shape[2]):
                    if conn[ii][jj][kk] != 0:
                        curnbr.append([ii, conn[ii][jj][kk] / np.max(conn[ii]), jj])
                    else:
                        curnbr.append([ii, 0.0, jj])
            all_nbrs.append(np.array(curnbr))

        all_nbrs = [sorted(nbrs, key=lambda x: x[1], reverse=True) for nbrs in all_nbrs]
        nbr_fea_idx = np.array([list(map(lambda x: x[2], nbr[:self.max_num_nbr])) for nbr in all_nbrs],
                               dtype=np.int64)
        nbr_fea = np.array([list(map(lambda x: x[1], nbr[:self.max_num_nbr])) for nbr in all_nbrs])
        nbr_fea = np.exp(-(nbr_fea[..., np.newaxis] - self.filter) ** 2 / self.step ** 2)
        return atom_fea, nbr_fea, nbr_fea_idx
