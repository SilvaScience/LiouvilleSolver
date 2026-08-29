import numpy as np

from projet_solver10 import (
    ExcitationSectorModel,
    FrequencyPathway,
    PropagationInterval,
    PureState,
    SpectroscopyProtocol,
    SpectroscopySolver,
)


def make_model():
    hamiltonians = {
        0: np.array([[0.0]]),
        1: np.array([[1.0, 0.08], [0.08, 1.2]]),
        2: np.array([[2.15]]),
    }
    raising = {
        (1, 0): np.array([[1.0], [0.6]]),
        (2, 1): np.array([[0.7, 1.1]]),
    }
    return ExcitationSectorModel(hamiltonians, raising)


def test_common_ground_and_adjoint_transition_blocks():
    model = make_model()

    assert model.sectors() == (0, 1, 2)
    assert [model.dimension(sector) for sector in model.sectors()] == [1, 2, 1]
    assert model.transition_decomposition() == "explicit_sector"

    initial = model.initial_condition()
    assert isinstance(initial, PureState)
    assert initial.sector == 0
    np.testing.assert_allclose(initial.vector, [1.0])

    j_10 = model.transition_blocks("J", "plus", 0)[1]
    j_01 = model.transition_blocks("J", "minus", 1)[0]
    j_21 = model.transition_blocks("J", "plus", 1)[2]
    j_12 = model.transition_blocks("J", "minus", 2)[1]
    np.testing.assert_allclose(j_01, j_10.conj().T)
    np.testing.assert_allclose(j_12, j_21.conj().T)


def test_dense_and_sparse_linear_responses_agree():
    model = make_model()
    pathway = FrequencyPathway(
        name="linear",
        interactions=("Ku",),
        component="linear",
    )
    protocol = SpectroscopyProtocol(
        intervals=(PropagationInterval("omega", "frequency", 1),),
        name="linear_frequency",
    )

    values = {}
    for backend in ("dense_liouville", "sparse_sector"):
        solver = SpectroscopySolver(backend=backend, eta=0.02)
        solver.feed_model(model)
        values[backend] = solver.calc_pathway(
            pathway, protocol, {"omega": 1.05}
        ).value

    np.testing.assert_allclose(
        values["sparse_sector"],
        values["dense_liouville"],
        rtol=1e-9,
        atol=1e-10,
    )
