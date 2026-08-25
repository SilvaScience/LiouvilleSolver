"""TeNPy finite-DMRG engine for static many-body references.

The engine translates SolverV10 spin-chain problems into TeNPy models, runs
finite-system DMRG, and returns ground-state energies and observables through
the common many-body result contract. TeNPy remains an optional dependency
loaded only when this engine is used.
"""

from __future__ import annotations

from time import perf_counter
import warnings

import numpy as np

from .many_body import (
    DMRGSettings,
    ManyBodyGroundStateResult,
    ManyBodySolver,
    SpinChainGroundStateProblem,
)


class TenpyDMRGEngine:
    """Charge-conserving finite DMRG for the open dimerized Heisenberg chain."""

    name = "tenpy_dmrg"

    @staticmethod
    def _imports():
        try:
            from tenpy.algorithms import dmrg
            from tenpy.models.spins import SpinChain
            from tenpy.networks.mps import MPS
        except ImportError as exc:
            raise ImportError(
                "TenpyDMRGEngine requires the optional package physics-tenpy."
            ) from exc
        return dmrg, SpinChain, MPS

    @staticmethod
    def _product_state(number_of_sites: int, total_sz: int) -> list[str]:
        state = ["up" if site % 2 == 0 else "down" for site in range(number_of_sites)]
        if total_sz == 0:
            return state
        if total_sz != 1:
            raise ValueError("The current gap contract supports total Sz=0 or 1.")
        for index in range(number_of_sites - 1, -1, -1):
            if state[index] == "down":
                state[index] = "up"
                return state
        raise RuntimeError("Could not construct the Sz=1 product state.")

    @staticmethod
    def _dmrg_options(settings: DMRGSettings) -> dict:
        return {
            "mixer": bool(settings.mixer),
            "max_E_err": float(settings.max_energy_error_eV),
            "max_sweeps": int(settings.max_sweeps),
            "trunc_params": {
                "chi_max": int(settings.chi_max),
                "svd_min": float(settings.svd_min),
            },
        }

    def _run_sector(self, model, settings: DMRGSettings, total_sz: int):
        dmrg, _, MPS = self._imports()
        psi = MPS.from_product_state(
            model.lat.mps_sites(),
            self._product_state(model.lat.N_sites, total_sz),
            bc=model.lat.bc_MPS,
            unit_cell_width=1,
        )
        start = perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*unused option.*",
            )
            info = dmrg.run(psi, model, self._dmrg_options(settings))
        runtime = perf_counter() - start
        energy = float(info["E"])
        sweep = info["sweep_statistics"]
        variance = float(np.real(model.H_MPO.variance(psi, energy)))
        diagnostics = {
            "runtime_seconds": runtime,
            "energy_variance_eV2": variance,
            "achieved_chi": int(max(psi.chi, default=1)),
            "physical_total_charge_2Sz": int(
                np.asarray(psi.get_total_charge(only_physical_legs=True)).ravel()[0]
            ),
            "sweeps": int(sweep["sweep"][-1] + 1),
            "final_energy_change_eV": float(sweep["Delta_E"][-1]),
            "final_entropy_change": float(sweep["Delta_S"][-1]),
            "final_norm_error": float(sweep["norm_err"][-1]),
            "final_max_truncation_error": float(sweep["max_trunc_err"][-1]),
            "maximum_entanglement_entropy": float(
                np.max(psi.entanglement_entropy())
            ),
        }
        return energy, psi, diagnostics

    @staticmethod
    def _bond_correlations(psi) -> np.ndarray:
        values = []
        for site in range(psi.L - 1):
            value = psi.expectation_value_term(
                [("Sz", site), ("Sz", site + 1)]
            )
            value += 0.5 * psi.expectation_value_term(
                [("Sp", site), ("Sm", site + 1)]
            )
            value += 0.5 * psi.expectation_value_term(
                [("Sm", site), ("Sp", site + 1)]
            )
            values.append(float(np.real(value)))
        return np.asarray(values)

    @staticmethod
    def _bulk_slice(number_of_bonds: int) -> slice:
        discard = number_of_bonds // 4
        stop = number_of_bonds - discard
        return slice(discard, max(discard + 1, stop))

    @staticmethod
    def _correlation_length_proxy(psi) -> float | None:
        center = psi.L // 2
        maximum_distance = min(psi.L // 3, psi.L - 1 - center)
        if maximum_distance < 4:
            return None
        sites = list(range(center + 2, center + maximum_distance + 1))
        correlations = np.asarray(
            psi.correlation_function(
                "Sz", "Sz", sites1=[center], sites2=sites
            )
        ).reshape(-1)
        distances = np.asarray(sites, dtype=float) - center
        amplitudes = np.abs(correlations)
        keep = amplitudes > max(1.0e-14, 1.0e-10 * float(np.max(amplitudes)))
        if np.count_nonzero(keep) < 3:
            return None
        slope, _ = np.polyfit(distances[keep], np.log(amplitudes[keep]), 1)
        return None if slope >= 0.0 else float(-1.0 / slope)

    @staticmethod
    def build_model(problem: SpinChainGroundStateProblem):
        """Build the shared TeNPy Hamiltonian used by static and dynamic runs."""
        _, SpinChain, _ = TenpyDMRGEngine._imports()
        couplings = np.asarray(
            [
                problem.J1_eV * (1.0 + (-1.0) ** site * problem.delta)
                for site in range(problem.number_of_sites - 1)
            ]
        )
        return SpinChain(
            {
                "L": problem.number_of_sites,
                "S": 0.5,
                "conserve": "Sz",
                "Jx": couplings,
                "Jy": couplings,
                "Jz": couplings,
                "bc_MPS": "finite",
                "bc_x": "open",
            }
        )

    def solve_state(
        self,
        problem: SpinChainGroundStateProblem,
        settings: DMRGSettings,
        total_sz: int = 0,
    ):
        """Return the optimized MPS for a dynamical engine without serializing it."""
        model = self.build_model(problem)
        energy, psi, diagnostics = self._run_sector(
            model, settings, total_sz=total_sz
        )
        return model, psi, energy, diagnostics

    def solve(
        self,
        problem: SpinChainGroundStateProblem,
        settings: DMRGSettings,
    ) -> ManyBodyGroundStateResult:
        model = self.build_model(problem)
        energy_sz0, psi0, diagnostics_sz0 = self._run_sector(
            model, settings, total_sz=0
        )
        energy_sz1, _, diagnostics_sz1 = self._run_sector(
            model, settings, total_sz=1
        )
        bonds = self._bond_correlations(psi0)
        signs = (-1.0) ** np.arange(bonds.size)
        bulk = self._bulk_slice(bonds.size)
        result = ManyBodyGroundStateResult(
            engine=self.name,
            problem=problem,
            settings=settings,
            energy_sz0_eV=energy_sz0,
            energy_sz1_eV=energy_sz1,
            gap_eV=energy_sz1 - energy_sz0,
            bond_correlations=bonds.tolist(),
            C1_all=float(np.mean(bonds)),
            D_all=float(np.mean(signs * bonds)),
            C1_bulk=float(np.mean(bonds[bulk])),
            D_bulk=float(np.mean((signs * bonds)[bulk])),
            correlation_length_proxy_sites=self._correlation_length_proxy(psi0),
            diagnostics={
                "Sz0": diagnostics_sz0,
                "Sz1": diagnostics_sz1,
                "bulk_bond_start": int(bulk.start),
                "bulk_bond_stop_exclusive": int(bulk.stop),
                "dependency": "physics-tenpy",
            },
        )
        if result.gap_eV < -10.0 * settings.max_energy_error_eV:
            raise RuntimeError(
                f"DMRG returned a negative sector gap {result.gap_eV:.3e} eV."
            )
        return result


ManyBodySolver.register_engine("tenpy_dmrg", TenpyDMRGEngine)
