from adaptive_qite import run_adaptive_qite
from _plots import make_pes_plots_per_param, make_entropy_plot
from utils import get_exact_fci_energy, make_geometry
from qiskit_nature.second_q.mappers import JordanWignerMapper
from state_and_hamiltonian import build_hamiltonian_and_state
import numpy as np

molecule = 'H2'
geometry= make_geometry(molecule)
basis_set =  'sto-3g'
charge = 0
spin = 0
n_activeorbs = 2
n_elec = 2
domain_size = 4
mapper = JordanWignerMapper()



## Fixed params for the optimization
total_time = 50
initial_dt = 0.05
max_theta = 0.05
min_dt = 0.0001
max_dt = 0.5
nsteps = int(total_time // initial_dt)

## Plot PES and entropy vs distance for H2
n_type = 1
energies_per_type = [[] for _ in range(n_type)]
energies_fci = []
timesteps_required = []
qcs = []

distances = np.arange(0.4, 3.5,0.1)

for distance in distances:
    geometry = f'H 0 0 0; H 0 0 {distance}'
    state, ham = build_hamiltonian_and_state(
        geometry,
        basis_set,
        n_elec,
        n_activeorbs,
        mapper
    )
    qc, energies_aqite, times, _, _ = run_adaptive_qite(
        ham,
        state,
        total_time,
        max_steps=nsteps,
        initial_dt=initial_dt,
        target_rotation=max_theta,
        max_dt=max_dt,
        min_dt=min_dt,
        domain_radius=domain_size,
        filename=f'{molecule}_out_init-dt={initial_dt}'
        )
    timesteps_required.append(len(energies_aqite))
    energy_aqite = energies_aqite[-1]
    energy_fci = get_exact_fci_energy(ham)
    energies_per_type[0].append(energy_aqite)
    energies_fci.append(energy_fci)
    qcs.append(qc)

# make sure to have figs/ folder in the same directory as main.py to save the plots
make_pes_plots_per_param(
    distances,
    [energies_per_type,],
    params=[1.0,],
    param_name = None,
    fci_energies=energies_fci,
    filename='H2_pes'
    )
make_entropy_plot(distances, qcs, filename='H2_entropy_vs_distance')