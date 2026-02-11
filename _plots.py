import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from qiskit.quantum_info import partial_trace, DensityMatrix, Statevector, entropy
import numpy as np
import matplotlib.ticker as ticker
from qiskit import QuantumCircuit


plt.rcParams.update({
    'font.size': 12,          # General font size
    'axes.labelsize': 12,     # X and Y labels
    'axes.titlesize': 12,     # Title
    'xtick.labelsize': 12,    # X-axis tick labels
    'ytick.labelsize': 12,    # Y-axis tick labels
    'legend.fontsize': 12,    # Legend
})
colors=[mcolors.TABLEAU_COLORS['tab:blue'],
        mcolors.TABLEAU_COLORS['tab:orange'],
        mcolors.TABLEAU_COLORS['tab:green'],
        mcolors.TABLEAU_COLORS['tab:red'],
        mcolors.TABLEAU_COLORS['tab:purple'],
        mcolors.TABLEAU_COLORS['tab:brown']]

def make_energy_vs_depth_plot(
        qcs_per_type: list[list[QuantumCircuit]],
        converged_energies_per_type: list[list[float]],
        dts: list[float],
        fci_energy: float,
        labels: list[str] = ['Fixed QITE', 'Adaptive QITE'],
        markers: list[str] = ['o', 'x'],
        filename: str ='figs/default_filename.pdf'
        ) -> None:
    
    """
    Saves a figure with converged energy vs circuit depth for different quantum algorithms (e.g. Adaptive QITE vs QITE)
    in a figs/ folder
        Args:
            - qcs_per_type: list, of N_types sublists, of len N_qcs_per_type, contains the quantum circuits obtained
            at convergence for each type of ansatz
            - converged_energies_per_type: list, of N_types sublists, of len N_qcs_per_type, contains the converged energy
            - dts: list, of len N_qcs_per_type, contains the initial dt used for each type of ansatz (e.g. for Adaptive
            QITE, the initial dt is the one used at the first iteration)
            - fci_energy: float, reference energy
            - labels: list, of N_types strings, where each string is the name of an ansatz
            - markers: list, of N_types strings, where each string is the marker associated to an ansatz
            - filename: str, name of the .pdf file to be saved
        Returns:
            Nothing
    """

    dt_color_map = dict(zip(dts, colors))

    plt.figure()
    marker_handles = [
    Line2D([0], [0], marker='o', markersize=8, color='w', label='Fixed QITE', linewidth=0,
           markerfacecolor='black'),
    Line2D([0], [0], marker='x', markersize=8, color='k', label='Adaptive QITE', linestyle=None, linewidth=0,
           markeredgewidth=1.5,
           markerfacecolor='black')
    ]
    color_handles = [
    Line2D([0], [0], marker='*', markersize=14, color='w', label=rf'(Initial) $dt$ = {dt}',
            markerfacecolor=c) for dt, c in dt_color_map.items()
    ]
    all_handles = marker_handles + color_handles

    plt.xlabel('Circuit depth')
    plt.ylabel('Converged energy [Ha]')
    plt.axhline(y = fci_energy, color='k', linestyle='--', label = f'Exact FCI energy at {fci_energy:.5f} Ha')

    for i, (qcs_type, converged_energies) in enumerate(zip(qcs_per_type, converged_energies_per_type)):
        depths = [qc.depth() for qc in qcs_type]
        for j, (depth, converged_energy) in enumerate(zip(depths, converged_energies)):
            plt.plot(depth, converged_energy, markersize=8, markeredgewidth=1.5, label=labels[i], marker=markers[i], color=colors[j], linestyle=None, alpha=0.8)
    plt.legend(handles = all_handles)
    plt.savefig('figs/' + filename + '.pdf', bbox_inches='tight')

def make_convergence_plots_per_param(
        iters: list[int],
        energies_per_type_per_param: list[list[list[float]]],
        params: list[float],
        param_name: str,
        fci_energy: float,
        labels: list[str] = ['Fixed QITE', 'Adaptive QITE'],
        markers: list[str] = ['o', '^'],
        filename: str = 'default_filename.pdf'
        ) -> None:
    
    """
    Saves a figure with (1 x 2) or (2 x N_params / 2) convergence subplots for different values of a given parameter in a figs/ folder
        Args:
            - iters: list, contains the integers [0, 1, 2, ..., N_iters-1]
            - energies_per_type_per_param: list, of N_params sublists, of N_types subsublists, of len N_iters, contains the energies
            at each value of the parameter, for each type of ansatz, and at each iteration
            - params: list, contains the different values of the parameter of interest
            - param_name: str, is the name of the parameter of interest
            - fci_energy: float, reference energy
            - labels: list, of N_types strings, where each string is the name of an ansatz
            - markers: list, of N_types strings, where each string is the marker associated to an ansatz
            - filename: str, name of the .pdf file to be saved
        Returns:
            Nothing
    """

    if len(params) <= 2:
        nrows = 1
        ncols = len(params)
    else:
        nrows = 2  # Fixed number of rows
        ncols = (len(params) + 1) // 2  # Calculate number of columns for 2 rows
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))  # Adjust figure size for m x n grid
    gs = fig.add_gridspec(nrows, ncols, wspace=0.1, hspace=0.3)  # m x n grid with spacing
    axs = gs.subplots(sharex=True, sharey=True)

    for i, param in enumerate(params):
        if len(params) <= 2:
            row = 0
            col = i
            ax = axs[i]
        else:
            row, col = divmod(i, ncols)  # Determine row and column for the grid
            ax = axs[row, col] if col > 1 else axs[row]  # Access subplot based on row and column
        ax.set_title(rf'{param_name} = {param}')  # Add title for each subplot
        ax.set_xlabel('Iterations')
        if col == 0:  # Add y-axis label only for the first column
            ax.set_ylabel('Energy (Ha)')

        for j, energies in enumerate(energies_per_type_per_param[i]):
            ax.plot(
                np.concatenate([iters, [len(iters),]])[::5], # plots every 5th iteration
                energies[50::3*5], # assuming 3 circuit evaluations per optimization step (SPSA with blocking)
                markersize=8,
                label=labels[j],
                alpha=0.7,
                linestyle=None,
                marker=markers[j],
                linewidth=0,
                color=colors[j]
                )
        ax.axhline(y=fci_energy, color='k', label=f'Exact FCI energy', linestyle='--')

    # Create a common legend
    handles, labels = axs.flat[-1].get_legend_handles_labels() if len(params) > 1 else axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15))
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')

def make_pes_plots_per_param(
        distances: list[float],
        energies_per_type_per_param: list[list[list[float]]], 
        params: list[float],
        param_name: str | None,
        fci_energies: list[float],
        labels: list[str] = ['Fixed QITE', 'Adaptive QITE'],
        markers: list[str] = ['o', '^'],
        filename: str = 'default_filename.pdf'
        ) -> None:
    
    """
    Saves a figure with 1 x N_params potential energy surface subplots for N_types different ansatze
    for different values of a given parameter in a figs/ folder
        Args:
            - distances: list, of N_dist float, contains each bond length at which the PES is evaluated
            - energies_per_type_per_param: list, of N_params sublists, of N_types subsublists, contains the energies at 
            each value of the parameter for each type of ansatz at each bond length
            - params: list, of N_params numbers, contains the different values of the parameter of interest
            - param_name: str, is the name of the parameter of interest
            - fci_energies: list, of N_dist numbers, contains the reference energy at each bond length
            - labels: list, of N_types strings, where each string is the name of an ansatz
            - markers: list, of N_types strings, where each string is the marker associated to an ansatz
            - filename: str, name of the .pdf file to be saved
        Returns:
            Nothing
    """

    fig = plt.figure()
    gs = fig.add_gridspec(1, len(params), wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)

    for i, param in enumerate(params):
        print('i', i)
        energies_per_type = [energies_per_type_per_param[i][j] for j in range(len(energies_per_type_per_param[i]))]
        ax = axs[i] if len(params) > 1 else axs
        if param_name is not None:
            ax.set_title(f'{param_name} =  {param}')  # Add title for each subplot
        ax.set_xlabel('Bond distance [Å]')
        if i == 0:
            ax.set_ylabel('Energy [Ha]')

        for j, energies in enumerate(energies_per_type):
            print('j', j)
            ax.plot(distances, energies, label=labels[j], marker=markers[j], alpha=0.7, markersize=8, markeredgewidth=1.5, linestyle=None, linewidth=0, color=colors[j])
        ax.plot(distances, fci_energies, label='Exact FCI', alpha=0.7, markersize=0, markeredgewidth=0, linestyle='--', color='k')

    # Create a common legend
    handles, labels = axs[-1].get_legend_handles_labels() if len(params) > 1 else axs.get_legend_handles_labels()
    print(handles, labels)
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15))
    plt.subplots_adjust(wspace=0)  # No horizontal space between subplots
    plt.tight_layout()
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')

def make_entropy_plot(
        distances: list[float],
        qcs: list[QuantumCircuit],
        filename: str = 'default_filename.pdf'
    ) -> None:

    """
    Saves a figure with a single subplot of the entanglement entropy of qubit 0 and 2 at N_dist
    different bond distances in a figs/ folder
        Args:
            - distances: list, of len N_dist, contains the bond lengths
            - qcs: list, of len N_dist, contains the converged states at the different bond lengths
            - filename: str, name of the .pdf file to be saved
        Returns:
            Nothing
    """  

    states = [Statevector(qc) for qc in qcs]
    rhos_q0 = [partial_trace(state, [1,3]) for state in states]
    entropies = [entropy(rho) for rho in rhos_q0]
    fig, ax = plt.subplots()
    ax.plot(distances, entropies, marker='o', label=r'Adaptive QITE', alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle='-.')
    ax.set_ylabel('Entanglement entropy')
    ax.set_xlabel('Bond distance [Å]')
    ax.legend(loc='best')
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')
    plt.close(fig)

def make_energy_dt_vs_iter_plot_general(
        energies_per_initdt: list[list[float]],
        dts_per_initdt: list[list[float]],
        init_dts: list[float],
        fci_energy: float,
        filename: str = 'default_filename'
) -> None:
    
    """
    Saves a figure with N subplots of energy and time intervals vs iterations for N different initial dts
    in a figs/ folder
        Args:
            - energies_per_initdt: list, of N sublists, of len N_iters, contains the energies at each iteration
            for each initial dt
            - dts_per_initdt: list, of N sublists, of len N_iters, contains the time intervals at each iteration
            - init_dts: list, of N floats, contains the different initial dts
            - fci_energy: float, reference energy
            - filename: str, name of the .pdf file to be saved
        Returns:
            Nothing
    """
    
    num_dts = len(init_dts)  # Determine the number of initial dts
    num_rows = 2  # Always use 2 rows
    num_cols = (num_dts + num_rows - 1) // num_rows  # Calculate columns needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2.5 * num_rows), sharex=False, sharey=True)
    axes = axes.flatten()  # Flatten axes for easier indexing

    prim_ax = axes[:num_dts]
    second_ax = [ax.twinx() for ax in prim_ax]

    for i, ax in enumerate(prim_ax):
        iters = np.arange(len(energies_per_initdt[i]))

        ax.set_title(r'$\Delta\tau^{(0)}$' + rf' = {init_dts[i]}')
        ax2_p = second_ax[i]
        ax.plot(iters, energies_per_initdt[i], label='Energy', marker='o', alpha=0.7, markersize=8, linestyle=None, linewidth=0, color='tab:blue')
        ax.axhline(fci_energy, color='k', linestyle='--', label='FCI Energy')

        ax2_p.plot(iters, dts_per_initdt[i], label='Time intervals', color='tab:orange', marker='^', alpha=0.7, markersize=8, linestyle=None, linewidth=0)
        
        # X labels only on bottom plots
        if i >= num_cols:
            ax.set_xlabel('Iterations')
        else:
            ax.set_xlabel('')
        
        # Y1 labels only on left plots
        if i % num_cols == 0:
            ax.set_ylabel('Energy')
        else:
            ax.set_ylabel('')
        
        # Y2 labels only on right plots
        if i % num_cols == num_cols - 1:
            ax2_p.set_ylabel('Time interval')
        else:
            ax2_p.set_ylabel('')

    # Hide unused subplots
    for j in range(num_dts, len(axes)):
        axes[j].axis('off')

    min_dt = min(min(dts) for dts in dts_per_initdt)
    max_dt = max(max(dts) for dts in dts_per_initdt)
    for ax2_p in second_ax:
        ax2_p.set_ylim(min_dt - 0.02, max_dt * 1.05)

    handles1, labels1 = axes[-1].get_legend_handles_labels() if len(init_dts) > 1 else axes.get_legend_handles_labels()
    handles2, labels2 = second_ax[-1].get_legend_handles_labels() if len(init_dts) > 1 else second_ax.get_legend_handles_labels()
    handles, labels = [handles1[0]] + handles2 + [handles1[1]], [labels1[0]] + labels2 + [labels1[1]]
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(bottom=0.15, hspace=0.4)
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')
    plt.close(fig)