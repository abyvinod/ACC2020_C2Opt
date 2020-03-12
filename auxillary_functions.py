import numpy as np
import matplotlib.pyplot as plt

skip_marker = 3
traj_markersize = 30
fig_fontsize = 10
fig_fontsize_legend = 9
fig_dpi = 150
fig_width = 3.4
fig_height = 2.8

params = {
   'axes.labelsize': fig_fontsize,
   'font.family': 'serif',
   'font.size': fig_fontsize,
   'legend.fontsize': fig_fontsize,
   'xtick.labelsize': fig_fontsize,
   'ytick.labelsize': fig_fontsize,
   'mathtext.fontset': 'cm',
   }
plt.rcParams.update(params)

def draw_initial_plot(xlim_tup, ylim_tup, target_position, cost_thresh,
                      initial_state, rand_init_traj_vec):
    """
    Plot the environment, distance contours, initial starting point, initial
    data, and the target set
    """
    # Draw the plot
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
    ax = fig.gca()
    ax.set_aspect('equal')                                 # Equal x and y axis
    plt.xlabel(r'$\mathrm{x}$', fontsize= 1 * fig_fontsize)
    plt.ylabel(r'$\mathrm{y}$', fontsize= 1 * fig_fontsize)
    ax.set_xlim(xlim_tup)
    ax.set_xticks(np.round(np.arange(xlim_tup[0], xlim_tup[1] + 1, 2)))
    ax.set_ylim(ylim_tup)
    plt.grid()

    draw_theta = np.linspace(0, 2 * np.pi, 100)         # For plotting circles
    zorder_init = 1e0                                   # Zorder for plotting
    
    # Draw contour plots
    for cost_val in (np.arange(1, 10) ** 2):
        r_temp = np.sqrt(cost_val * 2)
        plt.plot(target_position[0] + r_temp * np.cos(draw_theta),
                 target_position[1] + r_temp * np.sin(draw_theta),
                 color='k', linewidth=1)
        
    # Draw the initial state        
    plt.scatter(initial_state[0], initial_state[1], traj_markersize, 
            marker='d', color='y', label=r'$\mathrm{Initial\ state}$', 
            zorder=zorder_init + 2)
    
    plt.scatter(target_position[0], target_position[1], traj_markersize,
                marker = '*', color='r', zorder=11, 
                label=r'$\mathrm{Target\ state}$')
    
    # Draw the acceptable closeness to the target
    dist_thresh = np.sqrt(cost_thresh * 2)
    plt.plot(target_position[0] + dist_thresh * np.cos(draw_theta),
             target_position[1] + dist_thresh * np.sin(draw_theta),
             color='k', linestyle=':', linewidth=1, zorder=zorder_init,
             label=r'$\mathrm{Target\ set}$')
    
    # Draw initial trajectory
    plt.scatter(rand_init_traj_vec[1::skip_marker, 0], 
                rand_init_traj_vec[1::skip_marker, 1], traj_markersize, 
                marker='s', color='b', zorder=zorder_init, 
                label=r'$\mathrm{Initial\ data}$')
    
    # Interpolate the data points --- approximation
    plt.plot(rand_init_traj_vec[1:, 0], rand_init_traj_vec[1:, 1], color='b')
    plt.tight_layout()
    ax.legend(loc='upper left', ncol=1, prop={'size': 1 * fig_fontsize}, 
              labelspacing=0.25, framealpha=1)
    return ax
