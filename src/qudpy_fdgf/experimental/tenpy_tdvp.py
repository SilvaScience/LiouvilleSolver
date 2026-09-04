"""TeNPy two-site TDVP engine for connected spin correlations.

The engine prepares a DMRG ground state, applies a momentum-resolved spin
source, and evolves it in real time with two-site TDVP. It returns connected
correlation samples and diagnostics through the shared many-body dynamics
contract while keeping TeNPy optional.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .many_body_dynamics import (
    ManyBodyDynamicsSolver,
    RealTimeMPSSettings,
    SpinCorrelationProblem,
    SpinCorrelationResult,
)
from .tenpy_dmrg import TenpyDMRGEngine


class TenpyTDVPEngine:
    """Ground-state local/quasimomentum response without an MPDO."""

    name = "tenpy_tdvp"

    @staticmethod
    def _weights(number_of_sites: int, geometry: str) -> np.ndarray:
        if geometry == "local":
            weights = np.zeros(number_of_sites)
            weights[number_of_sites // 2] = 1.0
            return weights
        if geometry == "q0":
            return np.ones(number_of_sites) / np.sqrt(number_of_sites)
        if geometry == "qpi":
            return (-1.0) ** np.arange(number_of_sites) / np.sqrt(
                number_of_sites
            )
        raise ValueError(f"Unsupported operator geometry {geometry!r}.")

    @staticmethod
    def _operator_state(
        psi,
        weights: np.ndarray,
        operator: str,
        chi_max: int,
        svd_min: float,
    ):
        from tenpy.networks.mpo import MPO

        expectation = np.dot(
            weights, np.asarray(psi.expectation_value(operator))
        )
        active = np.flatnonzero(np.abs(weights) >= 1.0e-15)
        combined = psi.copy()
        if active.size == 1:
            site = int(active[0])
            combined.apply_local_op(
                site, operator, unitary=False, renormalize=False
            )
            coefficient = complex(weights[site])
            if coefficient != 1.0:
                combined = combined.add(
                    combined, alpha=0.0, beta=coefficient
                )
        else:
            wavepacket = MPO.from_wavepacket(
                psi.sites, weights, operator, eps=1.0e-15
            )
            wavepacket.apply(
                combined,
                {
                    "compression_method": "SVD",
                    "trunc_params": {
                        "chi_max": int(chi_max),
                        "svd_min": float(svd_min),
                    },
                },
            )
        if abs(expectation) > 1.0e-14:
            combined = combined.add(
                psi, alpha=1.0, beta=-expectation, cutoff=1.0e-14
            )
        return combined, expectation

    def correlate(
        self,
        problem: SpinCorrelationProblem,
        settings: RealTimeMPSSettings,
    ) -> SpinCorrelationResult:
        from tenpy.algorithms.tdvp import (
            SingleSiteTDVPEngine,
            TwoSiteTDVPEngine,
        )

        dmrg_engine = TenpyDMRGEngine()
        model, ground, energy, dmrg_diagnostics = dmrg_engine.solve_state(
            problem.ground_state, settings.dmrg, total_sz=0
        )
        weights = self._weights(
            problem.ground_state.number_of_sites,
            problem.operator_geometry,
        )
        times = settings.dt_eV_inverse * np.arange(
            settings.number_of_steps + 1, dtype=float
        )
        if (
            problem.operator_geometry == "q0"
            and problem.spin_component == "Sz"
        ):
            # O_q=0 is S^z_tot/sqrt(N).  The optimized state is in the exact
            # Sz_tot=0 charge sector, so the connected source is identically
            # zero; attempting to canonicalize a zero MPS is ill-defined.
            zeros = np.zeros(times.size)
            return SpinCorrelationResult(
                engine=self.name,
                problem=problem,
                settings=settings,
                times_eV_inverse=times.tolist(),
                correlation_real=zeros.tolist(),
                correlation_imag=zeros.tolist(),
                prepared_norm=0.0,
                ground_energy_eV=float(energy),
                diagnostics={
                    "runtime_seconds": 0.0,
                    "operator_expectation_real": 0.0,
                    "operator_expectation_imag": 0.0,
                    "maximum_achieved_chi": int(max(ground.chi, default=1)),
                    "final_achieved_chi": int(max(ground.chi, default=1)),
                    "maximum_entanglement_entropy": float(
                        np.max(ground.entanglement_entropy())
                    ),
                    "ground_state": dmrg_diagnostics,
                    "times_are_in_hbar_per_eV": True,
                    "symmetry_zero": "Sz_tot=0",
                },
            )
        prepared, expectation = self._operator_state(
            ground,
            weights,
            problem.spin_component,
            settings.chi_max,
            settings.svd_min,
        )
        detector = prepared.copy()
        prepared_norm_squared = float(
            max(0.0, np.real(prepared.overlap(prepared)))
        )
        correlations = np.zeros(times.size, dtype=complex)
        correlations[0] = prepared_norm_squared
        achieved_chi = [int(max(prepared.chi, default=1))]
        entropies = [
            float(np.max(prepared.entanglement_entropy()))
            if prepared.L > 1
            else 0.0
        ]
        start = perf_counter()
        if prepared_norm_squared > 1.0e-24:
            engine_class = (
                SingleSiteTDVPEngine
                if settings.tdvp_sites == 1
                else TwoSiteTDVPEngine
            )
            engine = engine_class(
                prepared,
                model,
                {
                    "dt": float(settings.dt_eV_inverse),
                    "N_steps": 1,
                    "max_dt": 1.01 * float(settings.dt_eV_inverse),
                    "trunc_params": {
                        "chi_max": int(settings.chi_max),
                        "svd_min": float(settings.svd_min),
                    },
                    "lanczos_params": {
                        "P_tol": 1.0e-12,
                    },
                },
            )
            for index in range(1, times.size):
                engine.run()
                correlations[index] = detector.overlap(prepared) * np.exp(
                    1.0j * energy * times[index]
                )
                achieved_chi.append(int(max(prepared.chi, default=1)))
                entropies.append(
                    float(np.max(prepared.entanglement_entropy()))
                )
        runtime = perf_counter() - start
        return SpinCorrelationResult(
            engine=self.name,
            problem=problem,
            settings=settings,
            times_eV_inverse=times.tolist(),
            correlation_real=np.real(correlations).tolist(),
            correlation_imag=np.imag(correlations).tolist(),
            prepared_norm=float(np.sqrt(prepared_norm_squared)),
            ground_energy_eV=float(energy),
            diagnostics={
                "runtime_seconds": runtime,
                "operator_expectation_real": float(np.real(expectation)),
                "operator_expectation_imag": float(np.imag(expectation)),
                "maximum_achieved_chi": int(max(achieved_chi)),
                "final_achieved_chi": int(achieved_chi[-1]),
                "maximum_entanglement_entropy": float(max(entropies)),
                "ground_state": dmrg_diagnostics,
                "times_are_in_hbar_per_eV": True,
            },
        )


ManyBodyDynamicsSolver.register_engine("tenpy_tdvp", TenpyTDVPEngine)
