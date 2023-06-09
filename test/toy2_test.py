# Loading modules
from __future__ import division
import numpy as np
import sys
import os

# Import MMA functions
from MMA import mmasub,subsolv,kktcheck

def main():

    m = 2
    n = 3

    epsimin = 0.0000001
    eeen = np.ones((n,1))
    eeem = np.ones((m,1))
    zeron = np.zeros((n,1))
    zerom = np.zeros((m,1))

    xval = np.array([[4,3,2]]).T
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = zeron.copy()
    xmax = 5*eeen
    low = xmin.copy()
    upp = xmax.copy()

    move = 1.0
    c = 1000*eeem
    d = eeem.copy()
    a0 = 1
    a = zerom.copy()

    outeriter = 0
    maxoutit = 11
    kkttol = 0	
	
    # Calculate function values and gradients of the objective and constraints functions
    if outeriter == 0:
        f0val,df0dx,fval,dfdx = toy2(xval)
        innerit = 0

    # The iterations starts
    kktnorm = kkttol+10
    outit = 0
    while (kktnorm > kkttol) and (outit < maxoutit):
        outit += 1
        outeriter += 1

        # The MMA subproblem is solved at the point xval:
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
            mmasub(m,n,outeriter,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,move)

        # Some vectors are updated:
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()

        # Re-calculate function values and gradients of the objective and constraints functions
        f0val,df0dx,fval,dfdx = toy2(xval)

        # The residual vector of the KKT conditions is calculated
        residu,kktnorm,residumax = \
            kktcheck(m,n,xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,xmin,xmax,df0dx,fval,dfdx,a0,a,c,d)




def toy2(xval):
    f0val = xval[0][0]**2+xval[1][0]**2+xval[2][0]**2
    df0dx = 2*xval
    fval1 = ((xval.T-np.array([[5, 2, 1]]))**2).sum()-9
    fval2 = ((xval.T-np.array([[3, 4, 3]]))**2).sum()-9
    fval = np.array([[fval1,fval2]]).T
    dfdx1 = 2*(xval.T-np.array([[5, 2, 1]]))
    dfdx2 = 2*(xval.T-np.array([[3, 4, 3]]))
    dfdx = np.concatenate((dfdx1,dfdx2))
    return f0val,df0dx,fval,dfdx


########################################################################################################
### RUN MAIN FUNCTION                                                                                ###
########################################################################################################

# Run main function / program
if __name__ == "__main__":
    main()
