import numpy as np
from numba import njit, float64, prange
from numba.experimental import jitclass
import matplotlib.pyplot as plt
from quantecon import MarkovChain, tauchen, rouwenhorst
from interpolation import interp
from scipy.linalg import eigvals
from scipy.special import roots_hermite
from scipy.optimize import fsolve, broyden1
import time

def gauss_hermite(n):
    "Calculates the Gauss-Hermite sample points and weights."
    points, weights = roots_hermite(n)
    points = np.sqrt(2) * points
    weights = weights / np.sqrt(np.pi)
    return points, weights

gh_pts, gh_wig = gauss_hermite(n=7)


cp_data = [
    ('β0', float64),             # the original discount factor
    ('β', float64),              # the transformed discount factor
    ('γ', float64),              # the relative risk aversion coefficient
    ('Rf0', float64),            # the original risk-free rate of return
    ('Rf', float64),             # the transformed risk-free rate of return
    ('θ', float64),              # the share of risky assets
    ('g', float64),              # the growth rate of income
    ('P', float64[:,:]),         # the transition probability of {Z_t}
    ('μZ', float64[:]),          # conditional mean of returns
    ('σZ', float64[:]),          # conditional std dev of returns
    ('YZ', float64[:]),          # the income vector
    ('gh_points', float64[:]),   # Gauss-Hermite sample points
    ('gh_weights', float64[:]),  # Gauss-Hermite weights
    ('grid_min', float64),       # minimum grid point for saving
    ('grid_med', float64),       # median grid point for saving
    ('grid_max', float64),       # maximum grid point for saving
    ('s_grid', float64[:])       # grid points for saving
]


@jitclass(cp_data)
class ConsumerProblem:
    "A class that stores primitives for the model economy."
    def __init__(self, 
                 β0=np.exp(-0.04/12),
                 γ=3.,
                 Rf0=np.exp(5.251e-4),
                 θ=0.6,
                 g=1.6213e-3,
                 P=np.array([[0.9854,0.0146],
                             [0.0902,0.9098]]),
                 μZ=np.array([6.8111e-3,-1.7201e-3]),
                 σZ=np.array([0.0383,0.0559]),
                 YZ=np.array([1,0.5]),
                 gh_points=gh_pts,
                 gh_weights=gh_wig,
                 grid_min=0.,
                 grid_med=10.,
                 grid_max=1e5,
                 grid_size=1000):
        
        self.β0, self.γ, self.Rf0, self.θ, self.g = β0, γ, Rf0, θ, g 
        self.P, self.μZ, self.σZ, self.YZ = P, μZ, σZ, YZ
        self.gh_points, self.gh_weights = gh_points, gh_weights
        self.grid_min, self.grid_med, self.grid_max = grid_min, grid_med, grid_max
        
        self.β = β0 * np.exp((1-γ)*g)
        self.Rf = Rf0 * np.exp(-g)
        
        # Construct exponential grid points for saving
        sp = (grid_med**2 - grid_min*grid_max)/(grid_min + grid_max - 2*grid_med)   # shift parameter
        s_grid = np.linspace(np.log(grid_min+sp), np.log(grid_max+sp), grid_size)
        self.s_grid = np.exp(s_grid) - sp
        self.s_grid[0] = grid_min
    
    def u_prime(self, x):
        "The marginal utility function."
        return x**(-self.γ)
    
    def u_prime_inv(self, x):
        "Inverse of the marginal utility function."
        return x**(-1/self.γ)

    
@njit(parallel=True)
def T(a_in,  # the asset level, float64[:,:]
      c_in,  # the consumption level, float64[:,:]
      cp):   # class with model information
    """
    The Coleman operator that updates the candidate consumption function 
    and the asset grid points via the endogenous grid method of Carroll (2006). 
    """
    β, Rf, θ = cp.β, cp.Rf, cp.θ
    P, μZ, σZ = cp.P, cp.μZ, cp.σZ
    YZ, s_grid = cp.YZ, cp.s_grid
    gh_points, gh_weights = cp.gh_points, cp.gh_weights
    u_prime, u_prime_inv = cp.u_prime, cp.u_prime_inv
    
    # Create candidate consumption function 
    def c_func(a, m):  # linear interpolation
        if a <= a_in[-1,m]:
            res = interp(a_in[:,m], c_in[:,m], a)
        else:          # linear extrapolation
            slope = (c_in[-1,m]-c_in[-2,m])/(a_in[-1,m]-a_in[-2,m])
            res = c_in[-1,m] + slope*(a-a_in[-1,m])
        return res
    
    # Create space to store updated consumption
    c_out = np.empty_like(c_in)
    
    # Calulate updated consumption
    for k in prange(len(s_grid)):
        s = s_grid[k]
        for j in prange(len(P)):
            # Compute expectation
            Ez = 0.
            for m in prange(len(P)):
                Y_hat = YZ[m]
                for n in prange(len(gh_points)):
                    R_hat = Rf*θ*np.exp(μZ[m] + σZ[m]*gh_points[n]) + Rf*(1-θ)
                    integrand = R_hat* u_prime(c_func(R_hat*s + Y_hat, m))
                    Ez += integrand * P[j,m] * gh_weights[n]
            c_out[k,j] = u_prime_inv(β * Ez)
    
    # Calculate the endogenous asset grid point
    a_out = np.empty_like(a_in)
    for m in prange(len(P)):
        a_out[:,m] = s_grid + c_out[:,m]
    
    # Fix a consumption-asset pair at (0,0) to improve interpolation
    c_out[0,:] = 0.
    a_out[0,:] = 0.
    
    return a_out, c_out


def solve_model_time_iter(model, oper, a_init, c_init, tol=1e-4, 
                          max_iter=10000, verbose=True, print_skip=500):
    """
    Time iteration using the endogenous grid method of Carroll (2006).
    ---------
    Returns :
    ---------
    a_new : the endogenous asset grid points
    c_new : the optimal consumption level
    k     : steps for the time iteration to terminite
    """
    k, err = 0, tol + 1
    
    while err>tol and k<max_iter:
        a_new, c_new = oper(a_init, c_init, model)
        #c_new = (c_init + c_new) / 2            # updating partially
        #err = np.max(np.abs(c_new - c_init))    # using absolute distance
        err = np.max(np.abs(c_new[1:,:]/c_init[1:,:] - 1))  # using relative distance
        a_init, c_init = np.copy(a_new), np.copy(c_new)
        k += 1
        
        if verbose and k%print_skip==0:
            print(f"Error at iteration {k} is {err}. ")
    
    if k == max_iter:
        print("Failed to converge!")
    
    if verbose and k < max_iter:
        print(f"\nConverged in {k} iterations.")
        
    return a_new, c_new, k


#@njit(parallel=True)
def K_α(cp, α):
    "Calculates the K(α) matrix, where K(α)(z,̂z) = P_{zẑ} E_{zẑ}β̂R̂^{α}."
    β, Rf, θ = cp.β, cp.Rf, cp.θ
    P, μZ, σZ = cp.P, cp.μZ, cp.σZ
    gh_points, gh_weights = cp.gh_points, cp.gh_weights
    
    expec = np.empty(len(P))
    
    for i in prange(len(P)):
        intg = (Rf*(θ*np.exp(μZ[i] + σZ[i]*gh_points) + 1-θ))**α
        expec[i] = np.sum(gh_weights * intg)
        
    Kα = β * (P * expec) 
    return Kα


def c_bar(cp):
    "Calculates the asymptotic marginal propensity to consume (AMPC)."
    γ = cp.γ
    K_mat = K_α(cp, 1-γ)
    spec = np.max(np.abs(eigvals(K_mat)))                 # spectral radius of K(1-γ)
    guess = (1 - spec**(1/γ)) * np.ones(len(cp.P))        # initial guess of AMPC
    
    opt = lambda x: 1/(1 + (K_mat @ x**(-γ))**(1/γ)) - x  # objective function
    cbar = fsolve(opt, guess)                             # AMPC
    return cbar


def c_bar2(cp, tol=1e-10, max_iter=50000):
    "An alternative way to calculate AMPC; by iterating F."
    γ = cp.γ
    K_mat = K_α(cp, 1-γ)
    spec = np.max(np.abs(eigvals(K_mat)))
    cbar_init =  (1 - spec**(1/γ)) * np.ones(len(cp.P))
    
    err, k = tol+1, 0
    while err>tol and k<max_iter:
        cbar_new = 1/(1 + (K_mat @ (cbar_init**(-γ)))**(1/γ))
        err = np.max(np.abs(cbar_new-cbar_init))
        cbar_init = cbar_new
        k += 1
    return cbar_new


#@njit(parallel=True)
def init_cand(cp, ampc, α=0):
    """
    An initial guess of the consumption function. (Method 1)
    c_0(a,z) = min{a, ̄c(z)a + E_z Y}
    """
    K, M = len(cp.s_grid), len(cp.P)
    a_init = np.empty((K,M))
    c_init = np.empty((K,M))
    
    for m in prange(M):
        a_init[:,m] = cp.s_grid + 1e-5
        a_vals = a_init[:,m]
        EY = np.sum(cp.P * cp.YZ, axis=1)  # or using EY = cp.YZ
        frac = α + (1-α)*ampc[m]
        c_init[:,m] = np.minimum(a_vals, frac*a_vals+ EY[m])
    return a_init, c_init


#@njit
def init_cand2(cp, ampc, α=0):
    """
    An initial guess of the consumption function. (Method 2)
    c_0(a,z) = min{a, ̄c(z)a + ̄a(z)[1-̄c(z)]}.
    Here ̄a(z):=(u')^{-1}[E_z βRu'[c(̂Y,̂Z)]] is the decision threshold.
    """
    β, Rf, θ = cp.β, cp.Rf, cp.θ
    P, YZ, μZ, σZ = cp.P, cp.YZ, cp.μZ, cp.σZ
    gh_points, gh_weights = cp.gh_points, cp.gh_weights
    
    # Approximate the decision threshold
    Z = len(P)
    μZ, σZ = np.atleast_2d(μZ).T, np.atleast_2d(σZ).T
    R = Rf * (θ*np.exp(μZ + σZ*gh_points) + 1-θ) 
    ER = np.sum(R * gh_weights, axis=1)
    expec = β * np.sum(P * (cp.u_prime(YZ) * ER), axis=1)
    thresh = cp.u_prime_inv(expec)  # a(z) ~ (u')^{-1}[E_z βRu'(̂Y)]
    
    # Calculate initial guess
    K, M = len(cp.s_grid), len(cp.P)
    a_init = np.empty((K,M))
    c_init = np.empty((K,M))
    
    for m in prange(M):
        a_init[:,m] = cp.s_grid + 1e-5
        a_vals = a_init[:,m]
        frac = α + (1-α)*ampc[m]
        linapp = frac*a_vals + (1-frac)*thresh[m]
        c_init[:,m] = np.minimum(a_vals, linapp)
    return a_init, c_init


def cal_mpc(cp, a_star, c_star):
    """
    Calculates the MPCs and the relative error of MPC vs AMPC.
    --------
    Inputs :
    --------
    cp : class with model information 
    a_star : the asset grid points endogenously determined by 
             the optimal consumption levels, float64[:,:]
    c_star : the optimal consumption levels, float64[:,:]
    ---------
    Outputs :
    ---------
    mpc : marginal propensity to consume, float64[:,:]
    err : the relative error of MPC vs AMPC, float64[:,:]
    ampc : asymptotic marginal propensity to consume, float64[:]
    """
    mpc = (c_star[1:,:]-c_star[:-1,:])/(a_star[1:,:]-a_star[:-1,:])
    ampc = c_bar(cp)           # AMPC
    err = np.abs(mpc/ampc -1)  # relative error
    return mpc, err, ampc


def speed_conv(alphas,          # parameter that controls the initial guess, float64[:]
               grid_sizes,      # number of grid points for asset/saving, int64
               grid_max=30.,    # maximum grid point for saving, float64
               grid_med=10.,    # median grid point for saving, float64
               tol=1e-5,        # tolerance level to terminate time iteration, float64
               verbose=True):
    """
    Calculates (i) the time taken to solve the optimal consumption function,
    and (ii) steps for the time iteration algorithm to terminate.
    (Note: the maximum grid point is fixed.)
    """
    step_conv = np.empty((len(alphas), len(grid_sizes)))  # steps to converge
    time_elap = np.empty_like(step_conv)  # time taken to solve the optimal consumption
    
    for i, α in enumerate(alphas):
        for j, grid_size in enumerate(grid_sizes):
            t_start = time.time()
            cp = ConsumerProblem(grid_med=grid_med, 
                                 grid_max=grid_max, 
                                 grid_size=grid_size)
            ampc = c_bar(cp)  # calculating the AMPC
            a_init, c_init = init_cand2(cp, ampc, α=α)  # initial guess, try also: init_cand
            a_star, c_star, kk = solve_model_time_iter(cp, T, a_init, c_init, 
                                                       tol=tol, verbose=False)
            time_elap[i,j] = time.time() - t_start
            step_conv[i,j] = kk
    
    if verbose:
        print(f'Iterations: \n{step_conv}')
        print(f'\nTime Taken: \n{time_elap}')
    return step_conv, time_elap


def speed_plots(α_space,       # parameter that controls the initial guess, float64[:]
                grid_sizes,    # number of grid points for asset/saving, int64
                grid_max,      # maximum grid point for saving, float64
                grid_med=10,   # median grid point for saving
                tol=1e-5,      # tolerance level to terminate time iteration
                figname=False):
    """
    Fixing the maximum grid point for saving, creates plots for: 
    (i) time taken to solve the optimal consumption function, and
    (ii) steps for the time iteration algorithm to terminate.
    """
    step_conv, time_elap = speed_conv(alphas=α_space, grid_sizes=grid_sizes, 
                                      grid_max=grid_max, grid_med=grid_med, 
                                      tol=tol, verbose=False)
    plt.subplots(figsize=(12,5))
    
    ax = plt.subplot(121)
    for i in range(len(grid_sizes)):
        ax.plot(α_space, step_conv[:,i], label=f'$G = {grid_sizes[i]}$')
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('Iterations')
    ax.legend()
    #plt.xscale('log')
    
    ax = plt.subplot(122)
    for i in range(len(grid_sizes)):
        ax.plot(α_space, time_elap[:,i], label=f'$G = {grid_sizes[i]}$')
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('Time (seconds)')
    #plt.xscale('log')
    
    if figname:
        plt.savefig(figname, bbox_inches='tight', pad_inches=0)  # dpi=500, 
    
    plt.show()


def speed_plots2(α_space,       # parameter that controls the initial guess, float64[:]
                 grid_sizes,    # number of grid points for asset/saving, int64
                 step_conv,
                 time_elap,
                 log_xscale=False,
                 add_grid=False,
                 figname=False):
    """
    Fixing the maximum grid point for saving, creates plots for: 
    (i) time taken to solve the optimal consumption function, and
    (ii) steps for the time iteration algorithm to terminate.
    """
    plt.subplots(figsize=(12,5))
    
    ax = plt.subplot(121)
    for i in range(len(grid_sizes)):
        ax.plot(α_space, step_conv[:,i], label=f'$G = {grid_sizes[i]}$')
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('Iterations')
    ax.legend()
    if add_grid:
        ax.grid(ls='--',color='grey')
    if log_xscale:
        plt.xscale('log')
    
    ax = plt.subplot(122)
    for i in range(len(grid_sizes)):
        ax.plot(α_space, time_elap[:,i], label=f'$G = {grid_sizes[i]}$')
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('Time (seconds)')
    if add_grid:
        ax.grid(ls='--',color='grey')
    if log_xscale:
        plt.xscale('log')
    
    if figname:
        plt.savefig(figname, bbox_inches='tight', pad_inches=0)  # dpi=500, 
    
    plt.show()


@njit(parallel=True)
def policy_eval(a_vals, a_star, c_star):
    """
    Evaluates the optimal consumption at different asset levels.
    --------
    Inputs :
    --------
    a_vals : asset levels for consumption evaluation, flaot64[:,:]
    a_star : asset grid points endogenously determined by 
             the optimal consumption levels, float64[:,:]
    c_star : the optimal consumption levels, float64[:,:]
    --------
    Output :
    --------
    c_vals : the level of consumption evaluated at a_vals, float64[:,:]
    """
    def c_func(a, m):  # linear interpolation
        if a <= a_star[-1,m]:
            res = interp(a_star[:,m], c_star[:,m], a)
        else:          # linear extrapolation
            slope = (c_star[-1,m]-c_star[-2,m])/(a_star[-1,m]-a_star[-2,m])
            res = c_star[-1,m] + slope*(a-a_star[-1,m])
        return res
    
    K, M = a_vals.shape
    c_vals = np.empty((K, M))
    for k in prange(K):
        for m in prange(M):
            c_vals[k,m] = c_func(a_vals[k,m], m)
    return c_vals

    
def err_func(cp, sG_min=10., tol=1e-5):
    """
    Computes the relative error of the truncated consumption function.
    --------
    Inputs :
    --------
    cp     : class with model information, used for solving the "true" 
             consumption function
             (Should set up a large saving space and fine grids.) 
    sG_min : the minimum s_G value, where s_G is the maximum grid point 
             for saving, float64
    tol    : the tolerance level to terminate the time iteration 
             algorithm, float64  
    --------
    Output :
    --------
    err : relative error of the truncated consumption function, float64[:]
    """
    ampc = c_bar(cp)                           # calculate the AMPC
    a_init, c_init = init_cand(cp, ampc, α=0)  # initial guess
    a_true, c_true, kk = solve_model_time_iter(cp, T, a_init, c_init, 
                                               tol=tol, verbose=False)  # the "true" consumption function
    idx = np.where(cp.s_grid<sG_min)[0][-1]    # the maximum index of the saving grid that is less than sG_min
                                               # i.e., index of the grid point that is closest to sG_min
    sG_space = cp.s_grid[idx:]  # the state space for sG
    err = np.empty(len(sG_space))
    err[-1] = 0.
    
    for i, gmax in enumerate(sG_space[:-1]):
        cp0 = dup_class(cp)
        cp0.grid_max, cp0.s_grid = gmax, cp.s_grid[:i+idx+1]
        a_init, c_init = init_cand(cp0, ampc, α=0)
        a_star, c_star, kk = solve_model_time_iter(cp0, T, a_init, c_init, 
                                                   tol=tol, verbose=False)
        c_vals = policy_eval(a_true, a_star, c_star)
        err[i] = np.max(np.abs(c_vals[1:,:]/c_true[1:,:] - 1))
        
    return err, sG_space
    
    
# --------------------------------------------------------- #
#            Some useful functions for computation
# ----------------------------------------------------------#
def exp_grids(gmin, gmax, gmed, N):
    """
    Constructs exponential grid points.
    --------
    Inputs :
    --------
    gmin : minimum grid point, float64
    gmax : maximum grid point, float64
    gmed : median grid point, float64
    --------
    Output :
    --------
    grids : the exponential grid points, float64[:]
    """
    s = (gmed**2 - gmin*gmax) / (gmin+gmax-2*gmed)
    grids = np.linspace(np.log(gmin+s), np.log(gmax+s), N)
    grids = np.exp(grids) - s
    if gmin == 0:
        grids[0] = 0
    return grids


def dup_class(cp):
    "Duplicates the class instance cp."
    β0, γ, Rf0, θ, g, P, μZ, \
    σZ, YZ, gh_points, gh_weights, \
    grid_min, grid_med, grid_max, size = cp.β0, cp.γ, cp.Rf0, cp.θ, cp.g, cp.P, \
                                         cp.μZ, cp.σZ, cp.YZ, cp.gh_points, cp.gh_weights, \
                                         cp.grid_min, cp.grid_med, cp.grid_max, len(cp.s_grid)
    
    cp_dup = ConsumerProblem(β0, γ, Rf0, θ, g, P, μZ, σZ, YZ,
                             gh_points, gh_weights, grid_min, grid_med, grid_max, size)
    return cp_dup