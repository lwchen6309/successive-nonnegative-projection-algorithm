import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from snpa import snpa


def create_time_resolved_spectra():
    t = np.linspace(0,50,50)
    v = np.linspace(0,40,100)
    trace = np.array([np.exp(-t/10), 1-np.exp(-t/10)]).T
    spectra = np.array([norm.pdf(v, 10, 1), norm.pdf(v, 25, 3)]).T
    spectra3d = trace.dot(spectra.T)
    return (t, trace), (v, spectra), spectra3d

def check_nmf(trspec, nmf_spectra, nmf_trace):
    """
    Check if not nan or inf in nmf_spectra and nmf_trace,
    and print error of decomposition.
    """
    
    if not all(np.isfinite(data).all() for data in [nmf_spectra, nmf_trace]):
        print('NMF error')
        mse, max_relerr = 0, 0
        return mse, max_relerr
    
    reconst_trspec = nmf_trace.dot(nmf_spectra)
    mse = np.sqrt(np.mean((reconst_trspec[:] - trspec[:])**2))
    eps = np.max(trspec[:]) * 1e-3
    rel_err = np.abs(1 - (reconst_trspec[:]+eps) / (trspec[:]+eps))
    max_relerr = np.max(rel_err)
    print('max relative error is %.4f'%max_relerr)
    return mse, max_relerr
    
def plot_spectra_compare(trspec, v, t, 
                        nmf_spectra, nmf_trace, 
                        ref_trace, time_index):
    
    fig, axes = plt.subplots(3,1)
    
    # NMF spectra
    axes[0].plot(v, nmf_spectra.T)
    legend_str = ['spectra_{nmf-sp %d}'%i 
                for i in range(nmf_spectra.shape[0])]
    axes[0].legend(legend_str)
    axes[0].set_title('NMF result')
    axes[0].set_xlabel('wavelength')
    
    # Time-resolved spectra
    axes[1].plot(v, trspec[time_index,:].T)
    legend_str = ['t = %.1f us'%t[i] for i in time_index]
    axes[1].legend(legend_str)
    axes[1].set_title('Original spectra')
    axes[1].set_xlabel('wavelength')
    
    # Temporal profile

    scale_ref_trace = ref_trace * \
        (np.max(nmf_trace,axis=0) / np.max(ref_trace,axis=0))
    axes[2].plot(t, nmf_trace)
    axes[2].plot(t, scale_ref_trace,'o')
    legend_str = ['trace_{nmf-sp %d}'%i 
                    for i in range(nmf_trace.shape[1])]
    legend_str.extend(['trace_{ref-peak %d}'%i 
                        for i in range(ref_trace.shape[1])])
    axes[2].legend(legend_str)
    axes[2].set_title('comparison of ref_trace and nmf_trace')
    axes[2].set_xlabel('time') 
    axes[2].set_ylabel('intensity')
    
    plt.subplots_adjust(hspace=0.5)
    
    return fig, axes

def plot_trspec(t, v, trspec):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    T, V = np.meshgrid(v, t)
    surf = ax.plot_surface(T, V, trspec, cmap=cm.jet)
    ax.set_title('time-resolved spectra')
    ax.set_xlabel('wavelength')
    ax.set_ylabel('time')
    ax.set_zlabel('intensity')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return ax
    
    
if __name__ == '__main__':
    plt.close('all')
    
    # Read excel file for spectral data of Criegee.
    (t, ref_trace), (v, ref_spectra), trspec = create_time_resolved_spectra()
    num_species = ref_spectra.shape[1]
    plt_trspec = plot_trspec(t, v, trspec)
    
    # Execute NMF
    [J, nmf_spectra] = snpa(trspec, num_species)
    nmf_trace = trspec[:, J]
    
    # Check NMF and correct by reference
    mse = check_nmf(trspec, nmf_spectra, nmf_trace)
    rot_spectra = np.linalg.pinv(ref_trace).dot(nmf_trace).dot(nmf_spectra)
    
    # Plot and save result
    time_index = np.linspace(0,trspec.shape[0]-1,4).astype(int)
    plot_spectra_compare(trspec,v,t,nmf_spectra,nmf_trace,ref_trace,time_index)
    # save_nmf_result(v,nmf_spectra,rot_spectra,t,nmf_trace,ref_trace)
    
    plt.show()