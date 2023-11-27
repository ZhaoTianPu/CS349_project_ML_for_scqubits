import scqubits as scq
import numpy as np

from sklearn.decomposition import PCA

from typing import List, Tuple, Union, Callable, Any, Optional
from typing_extensions import Literal


def tmon_sampler(
    param_list: List[float],
    normalize_energy: bool = True,
    evals_count: int = 4,
    ncut: int = 10,
) -> List[List[float] | List[np.ndarray]]:
    """
    Returns a list of excited state eigenenergies and eigenvectors of
    the transmon Hamiltonian for a given list of parameters.

    Parameters
    ----------
    param_list:
        list of transmon parameters, each entry is a list of the form [EC, EJ, ng]
    normalize_energy:
        if True, the energies are normalized by EC
    evals_count:
        number of energy levels to be calculated
    ncut:
        charge basis cutoff

    Returns
    -------
    List[np.ndarray | List[np.ndarray]]:
        list of eigenenergies and eigenvectors, each entry is a list of the form
        [[E1, E2, E3, ...], [eigenvector1, eigenvector2, eigenvector3, ...]]
    """
    tmon = scq.Transmon(EJ=30.0, EC=1.0, ng=0.3, ncut=ncut)
    evals_list = []
    evecs_list = []
    for params in param_list:
        tmon.EC = params[0]
        tmon.EJ = params[1]
        tmon.ng = params[2]
        evals, evecs = tmon.eigensys(evals_count=evals_count)
        if normalize_energy:
            evals /= tmon.EC
        evals_list.append(evals[1:])
        evecs_list.append(evecs[:, 1:].T)
    return [evals_list, evecs_list]


def process_tmon_data(
    param_list: List[float],
    data: List[List[float] | List[np.ndarray]],
    n_components: int = 10,
) -> Tuple[List[np.ndarray], PCA]:
    """
    Process the data returned by tmon_sampler() into a format that can be used
    by the model. All the eigenvectors are first processed with PCA, then for
    each example, the eigenvectors are flattened as a 1D array with length
    n_components*evals_count. Then, parameters, eigenenergies and eigenvectors
    are concatenated to form an 1D array with length
    3 + n_components*(evals_count-1)+evals_count-1.

    Parameters
    ----------
    param_list:
        list of transmon parameters, each entry is a list of the form [EC, EJ, ng]
    data:
        data returned by tmon_sampler()
    n_components:
        number of principal components to keep

    Returns
    -------
    Tuple[List[np.ndarray], PCA]:
        list of processed data (parameters, eigenenergies and eigenvectors)
        and the PCA object used to process the data
    """
    evals_list, evecs_list = data
    pca = PCA(n_components=n_components)
    # collect all eigenvectors
    all_evecs = np.concatenate(evecs_list)
    pca.fit(all_evecs)

    transformed_data = []
    for example_idx in range(len(evecs_list)):
        transformed_data.append(
            np.concatenate(
                (
                    param_list[example_idx],
                    evals_list[example_idx],
                    pca.transform(evecs_list[example_idx]).flatten(),
                )
            )
        )
    return transformed_data, pca
