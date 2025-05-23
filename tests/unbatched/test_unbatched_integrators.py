import pytest
import torch

import torch_sim as ts
from torch_sim.quantities import calc_kinetic_energy, calc_kT
from torch_sim.unbatched.models.lennard_jones import UnbatchedLennardJonesModel
from torch_sim.unbatched.unbatched_integrators import (
    MDState,
    calculate_momenta,
    npt_langevin,
    npt_nose_hoover,
    npt_nose_hoover_invariant,
    nve,
    nvt_langevin,
    nvt_nose_hoover,
    nvt_nose_hoover_invariant,
)
from torch_sim.units import MetalUnits


def test_nve_integrator(
    ar_supercell_sim_state: ts.SimState, unbatched_lj_model: UnbatchedLennardJonesModel
) -> None:
    """Test NVE integration conserves energy."""
    # Initialize integrator
    kT = torch.tensor(100.0) * MetalUnits.temperature  # Temperature in K
    dt = torch.tensor(0.001) * MetalUnits.time  # Small timestep for stability

    nve_init, nve_update = nve(model=unbatched_lj_model, dt=dt, kT=kT)

    # Remove batch dimension from cell
    ar_supercell_sim_state.cell = ar_supercell_sim_state.cell.squeeze(0)

    state = nve_init(state=ar_supercell_sim_state)

    # Run several steps
    energies = torch.zeros(20)
    for step in range(1000):
        state = nve_update(state, dt)
        if step % 50 == 0:
            total_energy = state.energy + calc_kinetic_energy(state.momenta, state.masses)
            energies[step // 50] = total_energy

    # Check energy conservation
    energy_drift = torch.abs(energies - energies[1]) / torch.abs(energies[1])
    assert torch.all(energy_drift < 0.05), "Energy should be conserved in NVE"


def test_nvt_langevin_integrator(
    ar_supercell_sim_state: ts.SimState, unbatched_lj_model: UnbatchedLennardJonesModel
) -> None:
    """Test Langevin thermostat maintains target temperature."""
    # Initialize integrator
    target_temp = torch.tensor(100.0) * MetalUnits.temperature
    dt = torch.tensor(0.001) * MetalUnits.time
    gamma = torch.tensor(10.0) / MetalUnits.time  # Friction coefficient

    langevin_init, langevin_update = nvt_langevin(
        model=unbatched_lj_model, dt=dt, kT=target_temp, gamma=gamma
    )

    # Remove batch dimension from cell
    ar_supercell_sim_state.cell = ar_supercell_sim_state.cell.squeeze(0)

    state = langevin_init(state=ar_supercell_sim_state, seed=42)
    # Run equilibration
    temperatures = torch.zeros(500)
    for step in range(500):
        state = langevin_update(state, target_temp)
        temp = calc_kT(state.momenta, state.masses) / MetalUnits.temperature
        temperatures[step] = temp

    average_temperature = torch.mean(temperatures)
    # Check temperature control
    assert 120 > average_temperature > 80, "Temperature should be maintained"


def test_nvt_nose_hoover_integrator(
    ar_supercell_sim_state: ts.SimState, unbatched_lj_model: UnbatchedLennardJonesModel
) -> None:
    """Test Nose-Hoover chain thermostat maintains temperature."""
    # Initialize integrator
    target_temp = torch.tensor(100.0) * MetalUnits.temperature
    dt = torch.tensor(0.001) * MetalUnits.time

    ar_supercell_sim_state.cell = ar_supercell_sim_state.cell.squeeze(0)

    nvt_init, nvt_update = nvt_nose_hoover(
        model=unbatched_lj_model,
        dt=dt,
        kT=target_temp,
        chain_length=3,
        chain_steps=3,
        sy_steps=3,
    )

    state = nvt_init(state=ar_supercell_sim_state, seed=42)

    # Run equilibration
    temperatures = torch.zeros(500)
    for step in range(500):
        state = nvt_update(state, target_temp)
        temp = calc_kT(state.momenta, state.masses) / MetalUnits.temperature
        temperatures[step] = temp

    average_temperature = torch.mean(temperatures)
    # Check temperature control
    assert 120 > average_temperature > 80, "Temperature should be maintained"

    # Check chain properties
    assert hasattr(state, "chain"), "Should have chain thermostat"
    assert hasattr(state.chain, "positions"), "Chain should have positions"
    assert hasattr(state.chain, "momenta"), "Chain should have momenta"
    assert state.chain.positions.shape[0] == 3, "Should have 3 chain thermostats"


def test_npt_langevin_integrator(
    ar_supercell_sim_state: ts.SimState, unbatched_lj_model: UnbatchedLennardJonesModel
) -> None:
    """Test Langevin thermostat maintains target temperature."""
    # Initialize integrator
    target_temp = torch.tensor(100.0) * MetalUnits.temperature
    dt = torch.tensor(0.001) * MetalUnits.time
    external_pressure = torch.tensor(10_000) * MetalUnits.pressure
    n_steps = 4000
    dim = ar_supercell_sim_state.positions.shape[1]
    langevin_init, langevin_update = npt_langevin(
        model=unbatched_lj_model,
        dt=dt,
        kT=target_temp,
        external_pressure=external_pressure,
        alpha=40 * dt,
    )

    # Remove batch dimension from cell
    ar_supercell_sim_state.cell = ar_supercell_sim_state.cell.squeeze(0)

    state = langevin_init(state=ar_supercell_sim_state, seed=42)
    # Run equilibration
    temperatures = torch.zeros(n_steps)
    pressures = torch.zeros(n_steps)
    for step in range(n_steps):
        state = langevin_update(state, target_temp)
        temp = calc_kT(state.momenta, state.masses) / MetalUnits.temperature
        volume = torch.linalg.det(state.cell)
        pressures[step] = (
            1
            / (dim)
            * (
                (2 * calc_kinetic_energy(state.momenta, state.masses) / volume)
                - torch.trace(state.stress)
            )
            / MetalUnits.pressure
        )
        temperatures[step] = temp

    average_temperature = torch.mean(temperatures[2000:])
    average_pressure = torch.mean(pressures[2000:])
    # Check temperature control
    assert 120 > average_temperature > 80, "Temperature should be maintained"
    assert 12000 > average_pressure > 8000, "Pressure should be maintained"


def test_integrator_state_properties(
    ar_supercell_sim_state: ts.SimState, unbatched_lj_model: UnbatchedLennardJonesModel
) -> None:
    """Test that all integrators preserve state properties."""
    device = ar_supercell_sim_state.positions.device
    dtype = ar_supercell_sim_state.positions.dtype

    momenta = calculate_momenta(
        ar_supercell_sim_state.positions,
        ar_supercell_sim_state.masses,
        100.0 * MetalUnits.temperature,
        device,
        dtype,
    )
    md_state = MDState(
        positions=ar_supercell_sim_state.positions,
        momenta=momenta,
        masses=ar_supercell_sim_state.masses,
        cell=ar_supercell_sim_state.cell.squeeze(0),
        pbc=ar_supercell_sim_state.pbc,
        forces=torch.zeros_like(ar_supercell_sim_state.positions),
        energy=torch.tensor(0.0),
        atomic_numbers=ar_supercell_sim_state.atomic_numbers,
    )

    for integrator in [nve, nvt_langevin, nvt_nose_hoover]:
        init_fn, update_fn = integrator(
            model=unbatched_lj_model,
            dt=torch.tensor(0.001) * MetalUnits.time,
            kT=torch.tensor(100.0) * MetalUnits.temperature,
        )
        state = init_fn(state=md_state)

        # Check basic state properties
        assert hasattr(state, "positions"), "Should have positions"
        assert hasattr(state, "momenta"), "Should have momenta"
        assert hasattr(state, "forces"), "Should have forces"
        assert hasattr(state, "masses"), "Should have masses"
        assert hasattr(state, "cell"), "Should have cell"
        assert hasattr(state, "pbc"), "Should have PBC flag"

        # Check tensor shapes
        n_atoms = len(state.masses)
        assert state.positions.shape == (n_atoms, 3)
        assert state.momenta.shape == (n_atoms, 3)
        assert state.forces.shape == (n_atoms, 3)
        assert state.masses.shape == (n_atoms,)
        assert state.cell.shape == (3, 3)

        assert torch.allclose(md_state.momenta, state.momenta)


def test_nvt_nose_hoover_invariant(
    ar_supercell_sim_state: ts.SimState, unbatched_lj_model: UnbatchedLennardJonesModel
) -> None:
    """Test Nose-Hoover chain thermostat maintains temperature."""
    # Initialize integrator
    target_temp = torch.tensor(100.0) * MetalUnits.temperature
    dt = torch.tensor(0.001) * MetalUnits.time

    ar_supercell_sim_state.cell = ar_supercell_sim_state.cell.squeeze(0)

    nvt_init, nvt_update = nvt_nose_hoover(
        model=unbatched_lj_model,
        dt=dt,
        kT=target_temp,
        chain_length=3,
        chain_steps=3,
        sy_steps=3,
    )

    state = nvt_init(state=ar_supercell_sim_state, seed=42)

    # Run equilibration
    invariant = torch.zeros(500)
    for step in range(500):
        state = nvt_update(state, target_temp)
        invariant[step] = nvt_nose_hoover_invariant(state, target_temp)

    # Check energy conservation
    invariant_drift = torch.abs(invariant - invariant[0]) / torch.abs(invariant[0])
    assert torch.all(invariant_drift < 0.05), "invariant should be conserved"


@pytest.mark.skip(reason="NPT Nose-Hoover needs debugging")
def test_npt_nose_hoover_invariant(
    ar_supercell_sim_state: ts.SimState, unbatched_lj_model: UnbatchedLennardJonesModel
) -> None:
    """Test NPT Nose-Hoover chain thermostats maintain temperature and pressure."""
    # Initialize integrator
    target_temp = torch.tensor(100.0) * MetalUnits.temperature
    target_pressure = torch.tensor(1.01325) * MetalUnits.pressure
    dt = torch.tensor(0.001) * MetalUnits.time

    ar_supercell_sim_state.cell = ar_supercell_sim_state.cell.squeeze(0)

    npt_init, npt_update = npt_nose_hoover(
        model=unbatched_lj_model,
        dt=dt,
        kT=target_temp,
        external_pressure=target_pressure,
        chain_length=3,
        chain_steps=3,
        sy_steps=3,
    )

    state = npt_init(state=ar_supercell_sim_state, seed=42)

    # Run equilibration
    invariant = torch.zeros(500)
    for step in range(500):
        state = npt_update(state, target_temp, target_pressure)
        invariant[step] = npt_nose_hoover_invariant(state, target_temp, target_pressure)

    # Check energy conservation
    invariant_drift = torch.abs(invariant - invariant[0]) / torch.abs(invariant[0])
    assert torch.all(invariant_drift < 0.05), "invariant should be conserved"
