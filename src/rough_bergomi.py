import numpy as np
import scipy.special as special


def volterra_BM_path_chol(grid_points, M, H, T,rho):
    """Volterra BM path Cholesky.

    Parameters
    -----------
    grid_points : int
        # points in the simulation grid
    H : float
        Hurst Index
    T : float
        time horizon
    M : int
        # paths to simulate
    """

    assert 0<H<1.0

    ## Step1: create partition

    X=np.linspace(0, T, num=grid_points)

    # get rid of starting point
    X=X[1:grid_points]

    ## Step 2: compute covariance matrix
    size=2*(grid_points-1)
    Sigma=np.zeros([size,size])
    #Sigma(1,1)
    for j in range(grid_points-1):
        for i in range(grid_points-1):
            if i==j:
                Sigma[i,j]=np.power(X[i],2*H)/2/H
            else:
                s=np.minimum(X[i],X[j])
                t=np.maximum(X[i],X[j])
                Sigma[i,j]=np.power(t-s,H-0.5)/(H+0.5)*np.power(s,0.5+H)*special.hyp2f1(0.5-H, 0.5+H, 1.5+H, -s/(t-s))
    #Sigma(1,2) and Sigma (2,1)
    for j in range(grid_points-1):
        for i in range(grid_points-1):
                Sigma[i,j+((grid_points-1))]=rho/(H+0.5)*(np.power(X[i],H+0.5)-np.power(X[i]-np.minimum(X[i],X[j]),H+0.5))
                Sigma[i+(grid_points-1),j]=rho/(H+0.5)*(np.power(X[j],H+0.5)-np.power(X[j]-np.minimum(X[i],X[j]),H+0.5))
    #Sigma(2,2)
    for j in range(grid_points-1):
        for i in range(grid_points-1):
                Sigma[i+(grid_points-1),j+(grid_points-1)]=np.minimum(X[i],X[j])

    ## Step 3: compute Cholesky decomposition
    P=np.linalg.cholesky(Sigma)

    ## Step 4: draw Gaussian rv

    Z=np.random.normal(loc=0.0, scale=1.0, size=[M,2*(grid_points-1)])

    ## Step 5: get (V,W) and add 0's in the beginning

    V=np.zeros((M,grid_points))
    W=np.zeros((M,grid_points))
    for i in range(M):
        aux=np.dot(P,Z[i,:])
        V[i,1:grid_points]=aux[0:(grid_points-1)]
        W[i,1:grid_points]=aux[(grid_points-1):2*(grid_points-1)]

    return V, W


def rough_bergomi(grid_points, M, H, T,rho,xi0,nu,S0):
    """Volterra BM path Cholesky.

    Parameters
    -----------
    grid_points : int
        # points in the simulation grid
    H : float
        Hurst Index
    T : float
        time horizon
    M : int
        # paths to simulate
    rho, xi0, nu, S0 : float
        Rough Bergomi model parameters
    """


    [V_path,W_path]=volterra_BM_path_chol(grid_points, M, H, T,rho)

    time_grid=np.linspace(0,T,grid_points)
    C_H=np.power(2*H*special.gamma(3.0/2.0-H)/special.gamma(H+0.5)/special.gamma(2.0-2.0*H),0.5)
    Variance=xi0*np.exp(2*nu*C_H*V_path-np.power(nu*C_H,2)*np.power(time_grid,2*H)/H)
    log_stock=np.ones([M,grid_points])*np.log(S0)
    time_increment=time_grid[1]
    brownian_increments=np.diff(W_path)
    for i in range(grid_points-1):
        log_stock[:,i+1]=log_stock[:,i]-0.5*Variance[:,i]*time_increment+np.power(Variance[:,i],0.5)*brownian_increments[:,i]


    return np.exp(log_stock)
