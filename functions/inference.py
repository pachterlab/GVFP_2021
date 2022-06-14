# Scientific computing imports
import numpy as np
from numpy.fft import irfftn
from numpy.random import choice
from scipy.stats import rv_discrete, poisson, nbinom, gaussian_kde
import matplotlib.pyplot as plt

# PyMC3-related imports
import pymc3 as pm
import theano.tensor as tt

# Plotting
import matplotlib.pyplot as plt
import arviz as az
import matplotlib as mpl

import loompy as lp

def convert_xy_to_params(x, y, beta, gamma, K_avg):
    kappa = (beta + gamma)*(x/(1-x))
    a_over_th = 1/y - 1
    theta = np.sqrt(kappa*K_avg/a_over_th)
    a = K_avg*kappa/theta
    return a, kappa, theta


# Sample from flattened 2D probability distribution p.
# mx = [mx0, mx1] is the shape of the 2D domain we're sampling.
def sample_from_p(mx, num_data_points, p_flat):
    x_N, x_M = np.arange(mx[0]), np.arange(mx[1])
    X_N, X_M = np.meshgrid(x_N, x_M, indexing='ij')    # get grid of possible values
    x_choices = np.arange(np.prod(mx))                 # represent each grid point as a point in a 1D vector
    
    samples = choice(x_choices, size=num_data_points, replace=True, p=p_flat)
    d_N, d_M = X_N.flatten()[samples], X_M.flatten()[samples]
    return d_N, d_M


# Get maximum of a 2D array
def get_2D_max(array):
    return np.unravel_index(array.argmax(), array.shape)


# Get KDE for smooth-looking heatmaps.
def get_2D_KDE(x_stats, y_stats, x_min=0, x_max=1, y_min=0, y_max=1):
    num_pts = 100                          # hyperparameter
    x_arg = np.linspace(x_min, x_max, num_pts)
    y_arg = np.linspace(y_min, y_max, num_pts)

    X_arg, Y_arg = np.meshgrid(x_arg, y_arg, indexing='ij')      # grid of points      X0, X1

 
    positions = np.vstack([X_arg.ravel(), Y_arg.ravel()])
    kernel = gaussian_kde([x_stats, y_stats])
    KDE = np.reshape(kernel(positions).T, X_arg.shape)
    return KDE, x_arg, y_arg

"""### Trivial model (constitutive/Poisson and mixture/NB) likelihood functions"""

# Constitutive model likelihood function.
def get_Poiss_2sp(mx, params):
    beta, gamma, a, kappa, theta = params
    K_avg = (a*theta)/kappa
    mu_N, mu_M = K_avg/beta, K_avg/gamma
    
    x_N, x_M = np.arange(mx[0]), np.arange(mx[1])
    X_N, X_M = np.meshgrid(x_N, x_M, indexing='ij')
    
    return poisson.pmf(X_N, mu_N)*poisson.pmf(X_M, mu_M)


# Mixture model likelihood function.
def get_NB_2sp(mx, params):
    beta, gamma, a, kappa, theta = params
    
    # Get generating function argument
    u = []
    half = mx[:]
    half[-1] = mx[-1]//2 + 1
    for i in range(len(mx)):
        l = np.arange(half[i])
        u_ = np.exp(-2j*np.pi*l/mx[i])-1
        u.append(u_)
    g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
    for i in range(len(mx)):
        g[i] = g[i].flatten()
    
    # Get generating function
    gf = np.exp(- (a/kappa)*np.log(1 - theta*(g[0]/beta + g[1]/gamma)))
    gf = gf.reshape(tuple(half))
                              
    Pss = irfftn(gf, s=mx)                        # Get Pss by inverse fast Fourier transform
    Pss = np.abs(Pss)/np.sum(np.abs(Pss))           # Normalize
    return Pss

"""### $\Gamma$-OU likelihood functions"""

# Get 2 species GOU generating function using ODE method
def get_gf_GOU_2sp_ODE(g0, g1, params):
    beta_0, beta_1, a, kappa, theta = params     # get parameters
    
    c0 = (g0) + (beta_0/(beta_1 - beta_0))*(g1)       #  relevant linear combinations of g_i
    c1 = - (beta_0/(beta_1 - beta_0))*(g1)
    
    min_fudge, max_fudge = 1, 10                                     # Determine integration time scale / length
    dt = np.min([1/kappa, 1/theta, 1/beta_0, 1/beta_1])*min_fudge
    t_max = np.max([1/kappa, 1/theta, 1/beta_0, 1/beta_1])*max_fudge
    num_tsteps = int(np.ceil(t_max/dt))
    
    t_array = np.linspace(0, t_max, num_tsteps+1)
    t_array = t_array.reshape((1, num_tsteps + 1))
    
    q = np.zeros((g0.shape[0], num_tsteps + 1), dtype=np.complex64)    # initialize array to store ODE
    c0 = c0.reshape((c0.shape[0],1))
    c1 = c1.reshape((c1.shape[0],1))
    q0 = theta*c0*(np.exp(-beta_0*t_array) - np.exp(-kappa*t_array))/(kappa - beta_0)
    q1 = theta*c1*(np.exp(-beta_1*t_array) - np.exp(-kappa*t_array))/(kappa - beta_1)
    q = q0 + q1
    

    integrand = q/(1-q)
    integral = np.trapz(integrand, dx=dt, axis=1)     # integrate ODE solution
    gf = np.exp( a*integral)               # get generating function
    return gf


# Get Pss for 2 species GOU model via ODE method
def get_GOU_2sp(mx, params):
    # Get generating function argument
    u = []
    half = mx[:]
    half[-1] = mx[-1]//2 + 1
    for i in range(len(mx)):
        l = np.arange(half[i])
        u_ = np.exp(-2j*np.pi*l/mx[i])-1
        u.append(u_)
    g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
    for i in range(len(mx)):
        g[i] = g[i].flatten()
    
    # Get generating function
    gf = get_gf_GOU_2sp_ODE(g[0], g[1], params)
    gf = gf.reshape(tuple(half))
    
    Pss = irfftn(gf, s=mx)                        # Get Pss by inverse fast Fourier transform
    Pss = np.abs(Pss)/np.sum(np.abs(Pss))           # Normalize
    return Pss


# Log likelihood of GOU model given data. Uses (x,y) as input instead of (kappa, theta).
def ll_GOU(phi, const, mx, data):
    # Get parameters
    x, y = phi
    beta, gamma, K_avg = const
    
    # Convert from (x, y) to original parameters
    a, kappa, theta = convert_xy_to_params(x, y, beta, gamma, K_avg)
    params = [beta, gamma, a, kappa, theta]
    
    Pss = get_GOU_2sp(mx, params)    # Compute Pss

    lp = np.log(Pss)
    result = np.sum(lp[data])
    return result


# Combines the above functions into one to reduce overhead associated with Python function calls.
# Helpful when doing expensive posterior sampling (since many likelihood function calls are required).
def ll_GOU2(phi, const, mx, data):
    # Get parameters
    x, y = phi
    beta, gamma, K_avg = const
    
    
    # Convert from (x, y) to original parameters
    a, kappa, theta = convert_xy_to_params(x, y, beta, gamma, K_avg)
    params = [beta, gamma, a, kappa, theta]
    
    
    # Get generating function argument
    u = []
    half = mx[:]
    half[-1] = mx[-1]//2 + 1
    for i in range(len(mx)):
        l = np.arange(half[i])
        u_ = np.exp(-2j*np.pi*l/mx[i])-1
        u.append(u_)
    g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
    for i in range(len(mx)):
        g[i] = g[i].flatten()
    
    # Get generating function
    beta_0 = beta
    beta_1 = gamma
    
    c0 = (g[0]) + (beta_0/(beta_1 - beta_0))*(g[1])       #  relevant linear combinations of g_i
    c1 = - (beta_0/(beta_1 - beta_0))*(g[1])
    
    min_fudge, max_fudge = 0.5, 10                                     # Determine integration time scale / length
    dt = np.min([1/kappa, 1/theta, 1/beta_0, 1/beta_1])*min_fudge
    t_max = np.max([1/kappa, 1/theta, 1/beta_0, 1/beta_1])*max_fudge
    num_tsteps = int(np.ceil(t_max/dt))
    
    t_array = np.linspace(0, t_max, num_tsteps+1)
    t_array = t_array.reshape((1, num_tsteps + 1))
    
    q = np.zeros((c0.shape[0], num_tsteps + 1), dtype=np.complex64)    # initialize array to store ODE
    c0 = c0.reshape((c0.shape[0],1))
    c1 = c1.reshape((c1.shape[0],1))
    q0 = theta*c0*(np.exp(-beta_0*t_array) - np.exp(-kappa*t_array))/(kappa - beta_0)
    q1 = theta*c1*(np.exp(-beta_1*t_array) - np.exp(-kappa*t_array))/(kappa - beta_1)
    q = q0 + q1

    integrand = q/(1-q)
    integral = np.trapz(integrand, dx=dt, axis=1)     # integrate ODE solution
    gf = np.exp( a*integral)               # get generating function
    
    gf = gf.reshape(tuple(half))
    
    Pss = irfftn(gf, s=mx)                        # Get Pss by inverse fast Fourier transform
    Pss = np.abs(Pss)/np.sum(np.abs(Pss))           # Normalize

    lp = np.log(Pss)
    result = np.sum(lp[data])
    return result

"""### CIR likelihood functions"""

# ODE for 2 species CIR model
def f_2sp(q, t, c0, c1, params):
    beta_0, beta_1, a, kappa, theta = params
    result = - kappa*q + theta*q*q + kappa*( c0*np.exp(-beta_0*t) + c1*np.exp(-beta_1*t)  )
    return result


# Vectorized RK4 implementation for 2 species CIR model
def RK4_2sp(q, f, t, c0, c1, step_size, param):
    j1 = f(q, t, c0, c1, param)
    j2 = f(q + (step_size/2)*j1, t + (step_size/2), c0, c1, param)
    j3 = f(q + (step_size/2)*j2, t + (step_size/2), c0, c1, param)
    j4 = f(q + (step_size)*j3, t + (step_size), c0, c1, param)
    
    q_new = q + (step_size/6)*(j1 + 2*j2 + 2*j3 + j4)
    return q_new


# Get 2 species CIR generating function using ODE method
def get_gf_CIR_2sp(g0, g1, params):
    beta_0, beta_1, a, kappa, theta = params     # get parameters
    
    c0 = (g0) + (beta_0/(beta_1 - beta_0))*(g1)       #  relevant linear combinations of g_i
    c1 = - (beta_0/(beta_1 - beta_0))*(g1)
    
    min_fudge, max_fudge = 0.5, 10                                     # Determine integration time scale / length
    dt = np.min([1/kappa, 1/theta, 1/beta_0, 1/beta_1])*min_fudge
    t_max = np.max([1/kappa, 1/theta, 1/beta_0, 1/beta_1])*max_fudge
    num_tsteps = int(np.ceil(t_max/dt))
     
    q = np.zeros((g0.shape[0], num_tsteps + 1), dtype=np.complex64)    # initialize array to store ODE
    
    # Solve ODE using RK4 method
    for i in range(0, num_tsteps):
        t = i*dt
        q[:,i+1] = RK4_2sp(q[:,i], f_2sp, t, c0, c1, dt, params)
        
    integral = np.trapz(q, dx=dt, axis=1)     # integrate ODE solution
    gf = np.exp((a*theta/kappa)*integral)               # get generating function
    return gf


# Get Pss for 2 species CIR model via ODE method
def get_CIR_2sp(mx, params):
    # Get generating function argument
    u = []
    half = mx[:]
    half[-1] = mx[-1]//2 + 1
    for i in range(len(mx)):
        l = np.arange(half[i])
        u_ = np.exp(-2j*np.pi*l/mx[i])-1
        u.append(u_)
    g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
    for i in range(len(mx)):
        g[i] = g[i].flatten()
    
    # Get generating function
    gf = get_gf_CIR_2sp(g[0], g[1], params)
    gf = gf.reshape(tuple(half))
                              
    Pss = irfftn(gf, s=mx)                        # Get Pss by inverse fast Fourier transform
    Pss = np.abs(Pss)/np.sum(np.abs(Pss))           # Normalize
    return Pss


# Log likelihood of CIR model given data. Uses (x,y) as input instead of (kappa, theta).
def ll_CIR(phi, const, mx, data):
    # Get parameters
    x, y = phi
    beta, gamma, K_avg = const
    
    # Convert from (x, y) to original parameters
    a, kappa, theta = convert_xy_to_params(x, y, beta, gamma, K_avg)
    params = [beta, gamma, a, kappa, theta]
    
    Pss = get_CIR_2sp(mx, params)    # Compute Pss

    lp = np.log(Pss)
    result = np.sum(lp[data])
    return result

"""### Bayes factor functions"""

# Generates synthetic data (either CIR or GOU) and computes log Bayes factor (e.g. P(CIR)/P(GOU))).
# Does this averaged over many trials (num_trials).
# Does this assuming different numbers of data points (e.g. num_data_points = [100, 200, 1000]).
def log_bayes_factor_avg(model, num_data_points, num_trials, mx, params):
    
    # Initialize log BF arrays
    num_kinds = len(num_data_points)
    log_bf_joint = np.zeros((num_kinds, num_trials))
    log_bf_nascent = np.zeros((num_kinds, num_trials))
    log_bf_mature = np.zeros((num_kinds, num_trials))
    
    
    # Get Pss for each model given this parameter set
    if model=='CIR':
        pss = get_CIR_2sp(mx, params)
        pss_other = get_GOU_2sp(mx, params)
    elif model=='GOU':
        pss = get_GOU_2sp(mx, params)
        pss_other = get_CIR_2sp(mx, params)
    p_flat = pss.flatten()
    
    
    # Get fake data
    for r in range(0, num_kinds):
        for t in range(0, num_trials):
            
            # Sample nascent and mature counts
            d_N, d_M = sample_from_p(mx, num_data_points[r], p_flat)

            # Log-likelihood functions
            lp = np.log10(pss)    # shape: x_N_domain by x_M_domain
            lp_nascent = np.log10(np.sum(pss, axis=1))
            lp_mature = np.log10(np.sum(pss, axis=0))

            lp_other = np.log10(pss_other)
            lp_other_nascent = np.log10(np.sum(pss_other, axis=1))
            lp_other_mature = np.log10(np.sum(pss_other, axis=0))

            # Log-likelihoods given data
            ll_joint = np.sum(lp[d_N, d_M])
            ll_nascent = np.sum(lp_nascent[d_N])
            ll_mature = np.sum(lp_mature[d_M])

            ll_other_joint = np.sum(lp_other[d_N, d_M])
            ll_other_nascent = np.sum(lp_other_nascent[d_N])
            ll_other_mature = np.sum(lp_other_mature[d_M])

            # Bayes factor
            log_bf_joint[r,t] = ll_joint - ll_other_joint
            log_bf_nascent[r,t] = ll_nascent - ll_other_nascent
            log_bf_mature[r,t] = ll_mature - ll_other_mature
                          
        log_bf_joint_avg = np.mean(log_bf_joint, axis=1)
        log_bf_nascent_avg = np.mean(log_bf_nascent, axis=1)
        log_bf_mature_avg = np.mean(log_bf_mature, axis=1)
    
    return log_bf_joint_avg, log_bf_nascent_avg, log_bf_mature_avg

# Compute averaged log Bayes factors for GOU vs Poisson, GOU vs NB, CIR vs Poisson, and CIR vs NB.
def log_bayes_factor_avg_null(num_data_points, num_trials, mx, params):
    
    # Get Pss for each model given this parameter set
    pss_CIR = get_CIR_2sp(mx, params)
    pss_GOU = get_GOU_2sp(mx, params)
    pss_Poiss = get_Poiss_2sp(mx, params)
    pss_NB = get_NB_2sp(mx, params)
    
    # Initialize log BF arrays
    log_bf_CIR_vs_Poiss, log_bf_CIR_vs_NB = np.zeros(num_trials), np.zeros(num_trials)
    log_bf_GOU_vs_Poiss, log_bf_GOU_vs_NB = np.zeros(num_trials), np.zeros(num_trials)
    
    for t in range(0, num_trials):
        
        # Get fake data
        d_GOU_N, d_GOU_M = sample_from_p(mx, num_data_points, pss_GOU.flatten())
        d_CIR_N, d_CIR_M = sample_from_p(mx, num_data_points, pss_CIR.flatten())


        # Log-likelihood functions
        lp_CIR = np.log10(pss_CIR)    # shape: mx[0] by mx[1]
        lp_GOU = np.log10(pss_GOU)
        lp_Poiss = np.log10(pss_Poiss)
        lp_NB = np.log10(pss_NB)


        # Log-likelihoods given data
        log_bf_CIR_vs_Poiss[t] = np.sum(lp_CIR[d_CIR_N, d_CIR_M]) - np.sum(lp_Poiss[d_CIR_N, d_CIR_M])
        log_bf_CIR_vs_NB[t] = np.sum(lp_CIR[d_CIR_N, d_CIR_M]) - np.sum(lp_NB[d_CIR_N, d_CIR_M])

        log_bf_GOU_vs_Poiss[t] = np.sum(lp_GOU[d_GOU_N, d_GOU_M]) - np.sum(lp_Poiss[d_GOU_N, d_GOU_M])
        log_bf_GOU_vs_NB[t] = np.sum(lp_GOU[d_GOU_N, d_GOU_M]) - np.sum(lp_NB[d_GOU_N, d_GOU_M])
        
    log_bf_CIR_vs_Poiss_avg = np.mean(log_bf_CIR_vs_Poiss)
    log_bf_CIR_vs_NB_avg = np.mean(log_bf_CIR_vs_NB)
    log_bf_GOU_vs_Poiss_avg = np.mean(log_bf_GOU_vs_Poiss)
    log_bf_GOU_vs_NB_avg = np.mean(log_bf_GOU_vs_NB)

    
    return log_bf_CIR_vs_Poiss_avg, log_bf_CIR_vs_NB_avg, log_bf_GOU_vs_Poiss_avg, log_bf_GOU_vs_NB_avg

"""### Bayesian parameter recovery functions"""

# This class is necessary for interfacing with PymC3.

class LogLike(tt.Op):
    
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)
    
    def __init__(self, const, mx, data, likelihood):
        
        # add inputs as class attributes
        self.const = const
        self.mx = mx
        self.data = data
        self.likelihood = likelihood
        
    def perform(self, node, inputs, outputs):
        
        phi, = inputs # this contains parmeters
        logl = self.likelihood(phi, self.const, self.mx, self.data) # calls the log-likelihood function
        outputs[0][0] = np.array(logl) # output the log-likelihood

# This function gets parameter posteriors via non-gradient-based sampling.
def get_parameter_posteriors(string_ID, const, mx, data, ll_func, draws_=2000, tune_=1000, chains_=None, cores_=None):
    
    # Parameter bounds
    epsilon = 0.005
    x_min, x_max = epsilon, 1-epsilon
    y_min, y_max = epsilon, 1-epsilon
    
    # Define log likelihood
    logl_op = LogLike(const, mx, data, ll_func)
    def logl_fun(phi):
        return logl_op(phi)
    
    # Define PyMC3 model
    model = pm.Model()
    with model:
        # Priors
        x_ = pm.Uniform('x', lower=x_min, upper=x_max)
        y_ = pm.Uniform('y', lower=y_min, upper=y_max)

        phi = tt.as_tensor_variable([x_, y_])

        # Likelihood
        pm.Potential('likelihood', logl_fun(phi))
        
        
    # Run PyMC3 model
    start_time = ti.time()
    with model:
        step = pm.DEMetropolisZ(tune = None)
        trace = pm.sample(draws = draws_, tune = tune_, step = step, chains = chains_, cores = cores_ )
    print("--- %s seconds ---" % (ti.time() - start_time))
        
    # Plot and save trace
    with model:
        axes = az.plot_trace(trace)
        fig = axes.ravel()[0].figure
        fig.savefig("results/pymc3_raw_"+string_ID+".png", bbox_inches='tight')
        fig.savefig("results/pymc3_raw_"+string_ID+".pdf", bbox_inches='tight')
        
    x_stats, y_stats = trace['x'], trace['y']
    return trace, x_stats, y_stats

# This function analyzes and plots the posterior samples.
def analyze_posteriors(string_ID, x_stats, y_stats, params_true, sim=True, loc=True, k_max=1, th_max=1):
    
    if sim:
        # True parameters
        beta, gamma, a_true, kappa_true, theta_true = params_true
        x_true, y_true = (kappa_true)/(kappa_true + beta + gamma), (theta_true)/(theta_true + a_true)
    else:
        # Get statistics in terms of kappa and theta
        beta, gamma, K_avg = params_true
        a_stats, kappa_stats, theta_stats = convert_xy_to_params(x_stats, y_stats, beta, gamma, K_avg)

    
    # KDEs
    KDE_xy, x_arg, y_arg = get_2D_KDE(x_stats, y_stats, 0, 1, 0, 1)
    if k_max==0:
        k_max = np.max(kappa_stats)
    if th_max==0:
        th_max = np.max(theta_stats)
    KDE_kth, k_arg, th_arg = get_2D_KDE(kappa_stats, theta_stats, 0, k_max, 0, th_max)
    
    KDE_x = np.sum(KDE_xy, axis=1)
    KDE_x = KDE_x/np.trapz(KDE_x, x=x_arg)
    KDE_y = np.sum(KDE_xy, axis=0)
    KDE_y = KDE_y/np.trapz(KDE_y, x=y_arg)
    
    KDE_k = np.sum(KDE_kth, axis=1)
    KDE_k = KDE_k/np.trapz(KDE_k, x=k_arg)
    KDE_th = np.sum(KDE_kth, axis=0)
    KDE_th = KDE_th/np.trapz(KDE_th, x=th_arg)
    
    # Summary statistics
    x_avg, y_avg = np.mean(x_stats), np.mean(y_stats)    # posterior means
    i, j = get_2D_max(KDE_xy)
    x_map, y_map = x_arg[i], y_arg[j]
    
    k_avg, th_avg = np.mean(kappa_stats), np.mean(theta_stats)    # posterior means
    i, j = get_2D_max(KDE_kth)
    k_map, th_map = k_arg[i], th_arg[j]
    
    
    
    
    print("Plotting histograms in terms of (x,y)...")
    vals, _,  _ = plt.hist(x_stats, alpha=0.5, density=True)
    # plt.plot(x_true*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='black', linewidth=2, label='true')
    plt.plot(x_avg*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='red', linewidth=2, label='avg')
    plt.plot(x_map*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='blue', linewidth=2, label='MAP')
    plt.plot(x_arg, KDE_x, color='black')
    plt.xlabel('$\\kappa/(\\kappa + \\beta + \\gamma)$', fontsize=30)
    plt.ylabel('Probability', fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig("results/post_marg_x_"+string_ID+".png", bbox_inches='tight')
    plt.savefig("results/post_marg_x_"+string_ID+".pdf", bbox_inches='tight')
    plt.show()

    vals, _, _ = plt.hist(y_stats, alpha=0.5, density=True)
    # plt.plot(y_true*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='black', linewidth=2, label='true')
    plt.plot(y_avg*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='red', linewidth=2, label='avg')
    plt.plot(y_map*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='blue', linewidth=2, label='MAP')
    plt.plot(y_arg, KDE_y, color='black')
    plt.xlabel('$\\theta/(\\theta + a)$', fontsize=30)
    plt.ylabel('Probability', fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig("results/post_marg_y_"+string_ID+".png", bbox_inches='tight')
    plt.savefig("results/post_marg_y_"+string_ID+".pdf", bbox_inches='tight')
    plt.show()
    
    
    print("Plotting histograms in terms of (kappa, theta)...")
    vals, _,  _ = plt.hist(kappa_stats, alpha=0.5, density=True)
    # plt.plot(kappa_true*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='black', linewidth=2, label='true')
    plt.plot(k_avg*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='red', linewidth=2, label='avg')
    plt.plot(k_map*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='blue', linewidth=2, label='MAP')
    plt.plot(k_arg, KDE_k, color='black')
    plt.xlabel('$\\kappa$', fontsize=30)
    plt.ylabel('Probability', fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig("results/post_marg_kappa_"+string_ID+".png", bbox_inches='tight')
    plt.savefig("results/post_marg_kappa_"+string_ID+".pdf", bbox_inches='tight')
    plt.show()

    vals, _, _ = plt.hist(theta_stats, alpha=0.5, density=True)
    # plt.plot(theta_true*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='black', linewidth=2, label='true')
    plt.plot(th_avg*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='red', linewidth=2, label='avg')
    plt.plot(th_map*np.ones(10), np.linspace(0, np.max(vals), 10), linestyle='--', color='blue', linewidth=2, label='MAP')
    plt.plot(th_arg, KDE_th, color='black')
    plt.xlabel('$\\theta$', fontsize=30)
    plt.ylabel('Probability', fontsize=30)
    plt.legend(fontsize=20)
    plt.savefig("results/post_marg_theta_"+string_ID+".png", bbox_inches='tight')
    plt.savefig("results/post_marg_theta_"+string_ID+".pdf", bbox_inches='tight')
    plt.show()
    
    s= mpl.rcParams['lines.markersize']**2
    newsize=3*s
    
    print("Plotting 2D heatmap in terms of (x, y)...")
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)

    ax.scatter(x_map, y_map, color='blue', label='MAP', s=newsize, edgecolor='white')
    ax.scatter(x_avg, y_avg, color='red', label='avg', s=newsize, edgecolor='white')
    # ax.scatter(x_true, y_true, color='black', label='true', s=newsize, edgecolor='white')
    ax.imshow(np.transpose(KDE_xy), origin='lower', extent=[0, 1, 0, 1])

    if loc==True:
        plt.legend(fontsize=18, framealpha=1)
    plt.savefig('results/post_2D_xy_'+string_ID+'.pdf', bbox_inches='tight')
    plt.savefig('results/post_2D_xy_'+string_ID+'.png', bbox_inches='tight')

    plt.show()
    
    
    
    
        
    print("Plotting 2D heatmap in terms of (kappa, theta)...")
    w1 = 60
    w2 = 60
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)

    ax.scatter(k_map, th_map, color='blue', label='MAP', s=newsize, edgecolor='white')
    ax.scatter(k_avg, th_avg, color='red', label='avg', s=newsize, edgecolor='white')
    # ax.scatter(kappa_true, theta_true, color='black', label='true', s=newsize, edgecolor='white')
    ax.imshow( np.transpose(KDE_kth[:w1,:w2]), origin='lower', extent=[0, k_arg[w1], 0, th_arg[w2]], aspect='auto')

    plt.savefig('results/post_2D_kth_'+string_ID+'.pdf', bbox_inches='tight')
    plt.savefig('results/post_2D_kth_'+string_ID+'.png', bbox_inches='tight')

    plt.show()

    return

"""### Real data Bayes factor functions"""
# This function gets parameter posteriors via smc.
def get_parameter_posteriors_smc(string_ID, const, mx, data, ll_func, draws_=1000):
    """"Arguments changed for sample_smc function. Here is pymc3 3.8"""
    # Parameter bounds
    epsilon = 0.005
    x_min, x_max = epsilon, 1-epsilon
    y_min, y_max = epsilon, 1-epsilon
    
    # Define log likelihood
    logl_op = LogLike(const, mx, data, ll_func)
    def logl_fun(phi):
        return logl_op(phi)
    
    # Define PyMC3 model
    model = pm.Model()
    with model:
        # Priors
        x_ = pm.Uniform('x', lower=x_min, upper=x_max)
        y_ = pm.Uniform('y', lower=y_min, upper=y_max)

        phi = tt.as_tensor_variable([x_, y_])

        # Likelihood
        pm.Potential('likelihood', logl_fun(phi))
        
        
    # Run PyMC3 model
    start_time = ti.time()
    with model:
        trace = pm.sample_smc(draws = draws_, cores = 1)
    #print("--- %s seconds ---" % (ti.time() - start_time))
        
    # Plot and save trace
    with model:
        axes = az.plot_trace(trace)
        fig = axes.ravel()[0].figure
        fig.savefig("results/pymc3_trace_"+string_ID+".png", bbox_inches='tight')
        #fig.savefig("results/pymc3_trace_"+string_ID+".pdf", bbox_inches='tight')
        
    return trace
  
def log_bayes_factor_data(args):
    gene_ID, data = args
    d_N, d_M = data
    mx = [np.max(d_N)+1,np.max(d_M)+1]
    K_avg = 1
    beta = K_avg/d_N.mean()
    gamma = K_avg/d_M.mean()
    const = [beta, gamma, K_avg]
    trace_GOU = get_parameter_posteriors_smc(gene_ID+"_GOU", const, mx, data, ll_func=ll_GOU2)
    trace_CIR = get_parameter_posteriors_smc(gene_ID+"_CIR", const, mx, data, ll_func=ll_CIR)
    lml_GOU = trace_GOU.report.log_marginal_likelihood
    lml_CIR = trace_CIR.report.log_marginal_likelihood
    log_BF = np.log10(np.exp(lml_GOU-lml_CIR))
    return log_BF, gene_ID

