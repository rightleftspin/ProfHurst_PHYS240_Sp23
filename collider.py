# Collider subroutine for DSMC

# import required modules
import numpy as np


def collider(v,crmax,dt,selxtra,coeff,sD) :
    """collider - Function to process collisions in cells
       Inputs
         v         Velocities of the particles
         crmax     Estimated maximum relative speed in a cell
         dt        Time step
         selxtra   Extra selections carried over from last timestep
         coeff     Coefficient in computing number of selected pairs
         sD        Object containing sorting lists
       Outputs
         col       Total number of collisions processed
    """

    ncell = sD.ncell
    col = 0              # Count number of collisions
    vrel = np.empty(3)   # Relative velocity for collision pair

    #* Loop over cells, processing collisions in each cell
    for jcell in range(ncell) :

        #* Skip cells with only one particle
        number = sD.cell_n[jcell]
        if number > 1:

            #* Determine number of candidate collision pairs
            #  to be selected in this cell
            select = coeff*number*(number-1)*crmax[jcell] + selxtra[jcell]
            nsel = int(select)            # Number of pairs to be selected
            selxtra[jcell] = select-nsel  # Carry over any left-over fraction
            crm = crmax[jcell]            # Current maximum relative speed

            #* Loop over total number of candidate collision pairs
            for isel in range(nsel) :

                #* Pick two particles at random out of this cell
                k = int(np.floor(np.random.uniform(0,number)))
                kk = int(np.ceil(k + np.random.uniform(0,number-1)) % number)
                ip1 = sD.Xref[k + sD.index[jcell]]   # First particle
                ip2 = sD.Xref[kk + sD.index[jcell]]  # Second particle

                #* Calculate pair's relative speed
                cr = np.linalg.norm(v[ip1,:] - v[ip2,:])   # Relative speed
                if cr > crm :         # If relative speed larger than crm,
                    crm = cr          # then reset crm to larger value

                #* Accept or reject candidate pair according to relative speed
                if cr/crmax[jcell] > np.random.random() :
                    #* If pair accepted, select post-collision velocities
                    col += 1                            # Collision counter
                    vcm = 0.5*(v[ip1,:] + v[ip2,:])   # Center of mass velocity
                    cos_th = 1 - 2*np.random.random()    # Cosine and sine of collision angle theta
                    sin_th = np.sqrt(1 - cos_th**2)
                    phi = 2*np.pi*np.random.random()       # Collision angle phi
                    vrel[0] = cr*cos_th                 # Compute post-collision relative velocity
                    vrel[1] = cr*sin_th*np.cos(phi)
                    vrel[2] = cr*sin_th*np.sin(phi)
                    v[ip1,:] = vcm + 0.5*vrel           # Update post-collision velocities
                    v[ip2,:] = vcm - 0.5*vrel
                    v[ip2,:] = vcm - 0.5*vrel

            crmax[jcell] = crm      # Update max relative speed

    return col