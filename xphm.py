import lal
import lalsimulation as lalsim
import numpy as np

def L0_func(params):
    """
    Computes the magnitude of the initial angular momentum L0.
    params: parameter dictionary
    """
    if params['mass_1'] < params['mass_2']:
        temp = params['mass_1']
        params['mass_1'] = params['mass_2']
        params['mass_2'] = temp
    m1 = params['mass_1']
    m2 = params['mass_2']
    M = m1 + m2
    eta = (m1*m2)/(M**2)
    s1z = params['chi_1z']
    s2z = params['chi_2z']
    fref = params['f_ref']

    Omega_o = np.pi*fref
    x = (M*Omega_o*lal.MTSUN_SI)**(2/3)

    delta = np.sqrt(1 - 4*eta)
    chi1L = s1z
    chi2L = s2z
    L0 = 1
    L1 = 3.0/2. + eta/6.0
    L2 = (81 + (-57 + eta)*eta)/24.
    L3 = (10935. + eta*(-62001 + 1674*eta + 7*eta**2 + 2214*np.pi**2))/1296.
    L32 = (-7*(chi1L + chi2L + chi1L*delta - chi2L*delta) + 5*(chi1L + chi2L)*eta)/6.
    L52 = (-1650*(chi1L + chi2L + chi1L*delta - chi2L*delta) + 1336*(chi1L + chi2L)*eta 
           + 511*(chi1L - chi2L)*delta*eta + 28*(chi1L + chi2L)*eta**2)/600. 

    PNexp = L0 + L1*x + L2*x**2 + L3*x**3 + L32*x**(3/2) + L52*x**(5/2)

    return eta*x**(-1/2)*PNexp

def thetaJL_func(params):
    """
    Computes thetaJL, the angle between the total angular momentum J and the initial angular momentum L0.
    params: parameter dictionary
    """
    J = J_from_params(params)
    Jz = J[2]
    return np.arccos(Jz/np.linalg.norm(J))

def phiJL_func(params):
    """
    Computes phiJL, the azimuthal angle of the total angular momentum J in the L0 frame.
    params: parameter dictionary
    """
    Jx, Jy, _ = J_from_params(params)
    return np.arctan2(Jy, Jx)


def phi_theta_from_J(Jx, Jy, Jz):
    norm = (Jx**2 + Jy**2 + Jz**2)**0.5
    return np.arctan2(Jy, Jx), np.arccos(Jz / norm)


def J_from_params(params):
    if params['mass_1'] < params['mass_2']:
        temp = params['mass_1']
        params['mass_1'] = params['mass_2']
        params['mass_2'] = temp
    m1 = params['mass_1']
    m2 = params['mass_2']
    M = m1 + m2
    s1x = params['chi_1x']
    s1y = params['chi_1y']
    s1z = params['chi_1z']
    s2x = params['chi_2x']
    s2y = params['chi_2y']
    s2z = params['chi_2z']

    # Components in L0 frame
    Jx = (m1/M)**2*s1x + (m2/M)**2*s2x
    Jy = (m1/M)**2*s1y + (m2/M)**2*s2y
    Jz = (m1/M)**2*s1z + (m2/M)**2*s2z + L0_func(params)
    J = np.array([Jx, Jy, Jz])
    return J


def rotate_z(V, theta):
    """
    Rotates the vector V by an angle theta around the z-axis.
    V: 1-D array with 3 components
    theta: angle
    """
    Vx = V[0]
    Vy = V[1]
    Vz = V[2]
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([Vx*c - Vy*s, Vx*s + Vy*c, Vz])

def rotate_y(V, theta):
    """
    Rotates the vector V by an angle theta around the y-axis.
    V: 1-D array with 3 components
    theta: angle
    """
    Vx = V[0]
    Vy = V[1]
    Vz = V[2]
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([Vx*c + Vz*s, Vy, -Vx*s + Vz*c])
    
def zeta_polarization(params):
    """
    Computes zeta, the polarization angle between the N frame where the x axis is aligned with L0's projection into the x-y plane and the spherical harmonic N frame where the y axis is aligned with J x N. This angle is needed to rotate the mode decomposition that we computed in the J frame into the frame aligned with L0 that convention dictates.
    params: parameter dictionary
    """
    thetaJN = params['thetaJN']
    kappa = params['kappa']
    thetaJL = thetaJL_func(params)
    
    # Components in J frame
    xp_x = np.cos(thetaJN)
    xp_y = 0.
    xp_z = -np.sin(thetaJN)
    xp = np.array([xp_x, xp_y, xp_z])
    
    yp_x = 0.
    yp_y = 1.
    yp_z = 0.
    yp = np.array([yp_x, yp_y, yp_z])
    
    N_x = np.sin(thetaJN)
    N_y = 0
    N_z = np.cos(thetaJN)
    N = np.array([N_x, N_y, N_z])
    
    L0_L0 = np.array([0, 0, 1]) # direction of L0 in L0 frame
    L0 = rotate_z(rotate_y(L0_L0, -thetaJL), -kappa)
    
    x = L0 - np.dot(L0, N)*N
    
    return np.arctan2(np.dot(x, yp), np.dot(x, xp))

def m2Ylm(l, m, thetaJN):
    """
    Computes the -2 spin weighted spherical harmonic evaluated at theta = thetaJN, phi = 0.
    l: l index
    m: m index
    thetaJN: thetaJN angle
    """
    m = int(m)
    l = int(l)
    return lal.SpinWeightedSphericalHarmonic(thetaJN, 0, -2, l, m) # kappa + np.pi

def wignerd_slow(l, m, mprime, beta):
    """
    Computes the wigner d matrix element with indices l, m, mprime, evaluated at angle beta. Only works for cases enumerated.
    l: l index
    m: m index
    mprime: mprime index
    beta: Euler angle beta
    """
    if mprime < 0:
        c = np.cos((np.pi - beta)/2)
        s = np.sin((np.pi - beta)/2)
        pf = (-1)**(l+m)
        mprime *= -1
    else:
        c = np.cos(beta/2)
        s = np.sin(beta/2)
        pf = 1
    return _wigner(l, m, mprime, c, s, pf)


def wignerd(l, m, mprime, eibetaover2):
    """
    Computes the wigner d matrix element with indices l, m, mprime, evaluated at angle beta. Only works for cases enumerated.
    l: l index
    m: m index
    mprime: mprime index
    eibetaover2: exp(i*beta/2)
    """
    if mprime < 0:
        c = np.imag(eibetaover2)
        s = np.real(eibetaover2)
        pf = (-1)**(l+m)
        mprime *= -1
    else:
        c = np.real(eibetaover2)
        s = np.imag(eibetaover2)
        pf = 1
    return _wigner(l, m, mprime, c, s, pf)


def _wigner(l, m, mprime, c, s, pf):
    if l == 2:
        if mprime == 2:
            if m == 2:
                f = c**4
            elif m == 1:
                f = 2*c**3*s
            elif m == 0:
                f = np.sqrt(6)*c**2*s**2
            elif m == -1:
                f = 2*c*s**3
            elif m == -2:
                f = s**4
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 1:
            if m == 2:
                f = -2*c**3*s
            elif m == 1:
                f = c**2*(c**2-3*s**2)
            elif m == 0:
                f = np.sqrt(6)*(c**3*s-c*s**3)
            elif m == -1:
                f = s**2*(3*c**2-s**2)
            elif m == -2:
                f = 2*c*s**3
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        else:
            raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    elif l == 3:
        if mprime == 3:
            if m == 3:
                f = c**6
            elif m == 2:
                f = np.sqrt(6)*c**6*s
            elif m == 1:
                f = np.sqrt(15)*c**4*s**2
            elif m == 0:
                f = 2*np.sqrt(5)*c**3*s**3
            elif m == -1:
                f = np.sqrt(15)*c**2*s**4
            elif m == -2:
                f = np.sqrt(6)*c*s**5
            elif m == -3:
                f = s**6
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 2:
            if m == 3:
                f = -np.sqrt(6)*c**5*s
            elif m == 2:
                f = c**4*(c**2-5*s**2)
            elif m == 1:
                f = np.sqrt(10)*c**3*(c**2*s-2*s**3)
            elif m == 0:
                f = np.sqrt(30)*c**2*s**2*(c**2-s**2)
            elif m == -1:
                f = np.sqrt(10)*s**3*(2*c**3-c*s**2)
            elif m == -2:
                f = s**4*(5*c**2-s**2)
            elif m == -3:
                f = np.sqrt(6)*c*s**5
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 1:
            f = 0
        else:
            raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    elif l == 4:
        if mprime == 4:
            if m == 4:
                f = c**8
            elif m == 3:
                f = 2*np.sqrt(2)*c**7*s
            elif m == 2:
                f = 2*np.sqrt(7)*c**6*s**2
            elif m == 1:
                f = 2*np.sqrt(14)*c**5*s**3
            elif m == 0:
                f = np.sqrt(70)*c**4*s**4
            elif m == -1:
                f = 2*np.sqrt(14)*c**3*s**5
            elif m == -2:
                f = 2*np.sqrt(7)*c**2*s**6
            elif m == -3:
                f = 2*np.sqrt(2)*c*s**7
            elif m == -4:
                f = s**8
            else:
                raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
        elif mprime == 3:
            f = 0
        elif mprime == 2:
            f = 0
        elif mprime == 1:
            f = 0
        else:
            raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    else:
        raise ValueError('Function wignerd not defined for (l, m, m\')=', l, m, mprime)
    return f*pf

def Atransfer_slow(l, m, mprime, alpha, beta, thetaJN):
    """
    Computes the mode-by-mode transfer function A defined in equation 3.7 of https://arxiv.org/pdf/2004.06503.pdf.
    l: l index
    m: m index
    mprime: mprime index
    alpha: Euler angle alpha
    beta: Euler angle beta
    thetaJN: angle thetaJN
    """
    return np.exp(-1j*m*alpha)*wignerd_slow(l, m, mprime, beta)*m2Ylm(l, m, thetaJN)


def Atransfer(l, m, mprime, eiabsmalpha, eibetaover2, thetaJN):
    """
    Computes the mode-by-mode transfer function A defined in equation 3.7 of https://arxiv.org/pdf/2004.06503.pdf.
    l: l index
    m: m index
    mprime: mprime index
    eiabsmalpha: exp(i*|m|*alpha)
    eibetaover2: exp(i*beta/2)
    thetaJN: angle thetaJN
    """
    if m > 0:
        eiabsmalpha = np.conj(eiabsmalpha)
    return np.conj(eiabsmalpha)*wignerd(l, m, mprime, eibetaover2)*m2Ylm(l, m, thetaJN)


def twist_factor_slow(l, mprime, alpha, beta, gamma, thetaJN, pol):
    """
    Computes the coefficients that twist the L frame modes into the plus and cross polarizations without the 1/2 or 1/2i. Seen in equations 3.5, 3.6 of https://arxiv.org/pdf/2004.06503.pdf.
    l: l index
    m: m index
    mprime: mprime index
    alpha: Euler angle alpha
    beta: Euler angle beta
    gamma: Euler angle gamma
    thetaJN: angle thetaJN
    pol: string, indicates polarization, accepted values are '+' and 'x'.
    """
    m_arr = np.arange(2*l+1)-l
    if pol == '+':
        summand = [Atransfer_slow(l, m, -mprime, alpha, beta, thetaJN) + (-1)**l*np.conj(Atransfer_slow(l, m, mprime, alpha, beta, thetaJN)) for m in m_arr]
    elif pol == 'x':
        summand = [Atransfer_slow(l, m, -mprime, alpha, beta, thetaJN) - (-1)**l*np.conj(Atransfer_slow(l, m, mprime, alpha, beta, thetaJN)) for m in m_arr]
    else:
        raise ValueError('Only + or x polarizations allowed.')
    return np.exp(1j*mprime*gamma)*np.sum(summand, axis=0)


def twist_factor(l, mprime, eimalpha, eibetaover2, eimprimegamma, thetaJN, pol):
    """
    Computes the coefficients that twist the L frame modes into the plus and cross polarizations without the 1/2 or 1/2i. Seen in equations 3.5, 3.6 of https://arxiv.org/pdf/2004.06503.pdf.
    l: l index
    m: m index
    mprime: mprime index
    eimalpha: exp(i*m*alpha) array, index 0 is m, ranges from 0 to max_l
    eibetaover2: exp(i*beta/2)
    eimprimegamma: exp(i*mprime*gamma)
    thetaJN: angle thetaJN
    pol: string, indicates polarization, accepted values are '+' and 'x'.
    """
    m_arr = np.arange(2*l+1)-l
    if pol == '+':
        summand = [Atransfer(l, m, -mprime, eimalpha[abs(m)], eibetaover2, thetaJN) + (-1)**l*np.conj(Atransfer(l, m, mprime, eimalpha[abs(m)], eibetaover2, thetaJN)) for m in m_arr]
    elif pol == 'x':
        summand = [Atransfer(l, m, -mprime, eimalpha[abs(m)], eibetaover2, thetaJN) - (-1)**l*np.conj(Atransfer(l, m, mprime, eimalpha[abs(m)], eibetaover2, thetaJN)) for m in m_arr]
    else:
        raise ValueError('Only + or x polarizations allowed.')
    return eimprimegamma*np.sum(summand, axis=0)

def h_Lframe(params, f_seq, l, m):
    """
    Computes the (l, m) L frame mode evaluated for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    l: l index
    m: m index
    """
    m = int(m)
    l = int(l)
    lal_dict_L = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMPrecModes(lal_dict_L,1); # 1 chooses L frame instead of J frame
    h_posf_L, _ = lalsim.SimIMRPhenomXPHMOneModeFrequencySequence(f_seq, l, m,
                                        params['mass_1'] * lal.MSUN_SI,
                                        params['mass_2'] * lal.MSUN_SI,
                                        params['chi_1x'],
                                        params['chi_1y'],
                                        params['chi_1z'],
                                        params['chi_2x'],
                                        params['chi_2y'],
                                        params['chi_2z'],
                                        params['distance'],
                                        0., #phiref doesn't matter
                                        params['f_ref'],
                                        lal_dict_L
                                       )
    return h_posf_L.data.data

def euler_angles(params, f_seq, mprime, backupNNLO = True):
    """
    Computes the (l, m) euler angles alpha, beta, and gamma evaluated for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    mprime: mprime index
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    """
    mprime = int(mprime)
    fs = f_seq.data
    phiz_of_f               = lal.CreateREAL8Sequence(len(fs))
    zeta_of_f               = lal.CreateREAL8Sequence(len(fs))
    costhetaL_of_f          = lal.CreateREAL8Sequence(len(fs))
    phiz_of_f.data          = np.zeros(len(fs))
    zeta_of_f.data          = np.zeros(len(fs))
    costhetaL_of_f.data     = np.zeros(len(fs))
    lalDict_MSA = lal.CreateDict();
    if not backupNNLO:
        # change flag to return an error if MSA fails
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lalDict_MSA, 222)
    lalsim.SimIMRPhenomXPMSAAngles(phiz_of_f, zeta_of_f, costhetaL_of_f, f_seq,
                               params['mass_1'] * lal.MSUN_SI,
                               params['mass_2'] * lal.MSUN_SI,
                               params['chi_1x'], params['chi_1y'], params['chi_1z'],
                               params['chi_2x'], params['chi_2y'], params['chi_2z'],
                               params['f_ref'], params['f_min'], params['f_max'], mprime,
                               lalDict_MSA);
    alpha   = phiz_of_f.data + np.pi - params['kappa']
    beta    = np.unwrap(np.arccos(costhetaL_of_f.data))
    epsilon = zeta_of_f.data # gamma = -epsilon
    return alpha, beta, -epsilon

def hpc_slow(params, f_seq, backupNNLO = True, modelist = np.array([[2,1], [2,2], [3,2], [3,3], [4,4]]) ):
    """
    Computes the plus and cross polarizations for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    modelist: array of modes to use (l, m)
    """
    thetaJN = params['thetaJN']
    n_freq = len(f_seq.data)
    hplus_J = np.zeros(n_freq, dtype=complex)
    hcross_J = np.zeros(n_freq, dtype=complex)
    mprime_arr = np.unique(modelist[:, 1])
    eulerangles = np.array([euler_angles(params, f_seq, mprime, backupNNLO) for mprime in mprime_arr])
    for mode in modelist:
        mpind = np.where(mprime_arr == mode[1])[0][0]
        hL = h_Lframe(params, f_seq, mode[0], -mode[1])
        hplus_J += 1/2*hL*twist_factor_slow(mode[0], mode[1], eulerangles[mpind, 0], eulerangles[mpind, 1], eulerangles[mpind, 2], thetaJN, '+')
        hcross_J += 1j/2*hL*twist_factor_slow(mode[0], mode[1], eulerangles[mpind, 0], eulerangles[mpind, 1], eulerangles[mpind, 2], thetaJN, 'x')
    zeta = zeta_polarization(params)
    hplus = np.cos(2*zeta)*hplus_J + np.sin(2*zeta)*hcross_J
    hcross = np.cos(2*zeta)*hcross_J - np.sin(2*zeta)*hplus_J
    return hplus, hcross


def create_eimalpha(alpha, max_l):
    eialpha = np.exp(1j*alpha)
    if max_l == 2:
        return np.array([0*alpha+1, eialpha, eialpha*eialpha])
    elif max_l == 3:
        eialpha2 = eialpha*eialpha
        return np.array([0*alpha+1, eialpha, eialpha2, eialpha2*eialpha])
    elif max_l == 4:
        eialpha2 = eialpha*eialpha
        return np.array([0*alpha+1, eialpha, eialpha2, eialpha2*eialpha, eialpha2*eialpha2])
    else:
        raise ValueError('This code currently only supports modes up to mprime=4, l=4 and requires the quadrupole mode.')

def eintheta(theta, n):
    eitheta = np.exp(1j*theta)
    if n == 0:
        return 0*eitheta + 1
    elif n == 1:
        return eitheta
    elif n == 2:
        return eitheta*eitheta
    elif n == 3:
        return eitheta*eitheta*eitheta
    elif n == 4:
        eitheta2 = eitheta*eitheta
        return eitheta2*eitheta2
    
def create_eimprimegamma(gamma, mprime_arr):
    return np.array([eintheta(gamma[i], mprime_arr[i]) for i in range(len(mprime_arr))])

def hpc(params, f_seq, backupNNLO = True, modelist = np.array([[2,1], [2,2], [3,2], [3,3], [4,4]]) ):
    """
    Computes the plus and cross polarizations for the parameters in the dict params at the frequencies in f_seq more quickly.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    modelist: array of modes to use (l, m)
    """
    thetaJN = params['thetaJN']
    n_freq = len(f_seq.data)
    hplus_J = np.zeros(n_freq, dtype=complex)
    hcross_J = np.zeros(n_freq, dtype=complex)
    mprime_arr = np.unique(modelist[:, 1])
    eulerangles = np.array([euler_angles(params, f_seq, mprime, backupNNLO) for mprime in mprime_arr])
    max_l = np.max(modelist[:,0])
    eimalpha = create_eimalpha(eulerangles[:, 0], max_l)
    eibetaover2 = np.exp(eulerangles[:, 1]*1j/2)
    eimprimegamma = create_eimprimegamma(eulerangles[:, 2], mprime_arr)
    for mode in modelist:
        mpind = np.where(mprime_arr == mode[1])[0][0]
        hL = h_Lframe(params, f_seq, mode[0], -mode[1])
        hplus_J += 1/2*hL*twist_factor(mode[0], mode[1], eimalpha[:, mpind], eibetaover2[mpind], eimprimegamma[mpind], thetaJN, '+')
        hcross_J += 1j/2*hL*twist_factor(mode[0], mode[1], eimalpha[:, mpind], eibetaover2[mpind], eimprimegamma[mpind], thetaJN, 'x')
    zeta = zeta_polarization(params)
    hplus = np.cos(2*zeta)*hplus_J + np.sin(2*zeta)*hcross_J
    hcross = np.cos(2*zeta)*hcross_J - np.sin(2*zeta)*hplus_J
    return hplus, hcross

def hpc_component(params, f_seq, l, mprime, backupNNLO = True):
    """
    Computes the (l, mprime) component of plus and cross polarizations for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    l: l index
    mprime: mprime index
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    """
    thetaJN = params['thetaJN']
    n_freq = len(f_seq.data)
    hplus_J = np.zeros(n_freq, dtype=complex)
    hcross_J = np.zeros(n_freq, dtype=complex)
    eulerangles = euler_angles(params, f_seq, mprime, backupNNLO)
    hL = h_Lframe(params, f_seq, l, -mprime)
    hplus_J = 1/2*hL*twist_factor_slow(l, mprime, eulerangles[0], eulerangles[1], eulerangles[2], thetaJN, '+')
    hcross_J = 1j/2*hL*twist_factor_slow(l, mprime, eulerangles[0], eulerangles[1], eulerangles[2], thetaJN, 'x')
    zeta = zeta_polarization(params)
    hplus = np.cos(2*zeta)*hplus_J + np.sin(2*zeta)*hcross_J
    hcross = np.cos(2*zeta)*hcross_J - np.sin(2*zeta)*hplus_J
    return hplus, hcross

def hpc_component_fast(params, f_seq, l, mprime, backupNNLO = True):
    """
    DOES NOT YET WORK!

    Computes the (l, mprime) component of plus and cross polarizations for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    l: l index
    mprime: mprime index
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    """
    thetaJN = params['thetaJN']
    n_freq = len(f_seq.data)
    hplus_J = np.zeros(n_freq, dtype=complex)
    hcross_J = np.zeros(n_freq, dtype=complex)
    eulerangles = euler_angles(params, f_seq, mprime, backupNNLO)
    eimalpha = create_eimalpha(eulerangles[0], l)
    eibetaover2 = np.exp(eulerangles[1]*1j/2)
    eimprimegamma = create_eimprimegamma(eulerangles[2], mprime)
    hL = h_Lframe(params, f_seq, l, -mprime)
    hplus_J += 1/2*hL*twist_factor(l, mprime, eimalpha, eibetaover2, eimprimegamma, thetaJN, '+')
    hcross_J += 1j/2*hL*twist_factor(l, mprime, eimalpha, eibetaover2, eimprimegamma, thetaJN, 'x')
    zeta = zeta_polarization(params)
    hplus = np.cos(2*zeta)*hplus_J + np.sin(2*zeta)*hcross_J
    hcross = np.cos(2*zeta)*hcross_J - np.sin(2*zeta)*hplus_J
    return hplus, hcross


def lal_hpc(params, f_seq):
    """
    Computes the plus and cross polarizations for the parameters in the dict params at the frequencies in f_seq by calling lal for the polarizations directly.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    """
    pdict = {'DL': params['distance'],
                 'phiref': params['phi_ref'],
                 'f_ref': params['f_ref'],
                 'inclination': params['inclination'],
                 'm1': params['mass_1'] * lal.MSUN_SI, 'm2': params['mass_2'] * lal.MSUN_SI,
                 's1x': params['chi_1x'], 's1y': params['chi_1y'], 's1z': params['chi_1z'],
                 's2x': params['chi_2x'], 's2y': params['chi_2y'], 's2z': params['chi_2z']}

    PARAMNAMES = ['phiref', 'm1', 'm2', 's1x', 's1y', 's1z',
                          's2x', 's2y', 's2z', 'f_ref', 'DL', 'inclination']

    # Tidal deformabilities are zero for black holes
    lal_pars = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(lal_pars, 0.)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(lal_pars, 0.)

    # Parameters in the order that LAL takes, give approximant for the IMRPhenomXPHM model
    wfparams = [pdict[p] for p in PARAMNAMES] \
        + [lal_pars, lalsim.GetApproximantFromString("IMRPhenomXPHM"), f_seq]

    # Generate hplus, hcross
    hplus, hcross = lalsim.SimInspiralChooseFDWaveformSequence(*wfparams)

    return hplus.data.data, hcross.data.data


def compute_response_coeffs(params, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes the detector response coefficients for the parameters in the dict params.
    params: parameter dictionary
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    ra = params['ra']
    dec = params['dec']
    psi = params['psi']
    tgps = params['tgps']

    ndet = len(detstrings)
    
    det_response = [lal.CachedDetectors[DET_CODE[det_name]].response
                                 for det_name in detstrings]

    gmst = lal.GreenwichMeanSiderealTime(tgps)
    gha = gmst - ra

    X = np.array([-np.cos(psi)*np.sin(gha)-np.sin(psi)*np.cos(gha)*np.sin(dec),
                 -np.cos(psi)*np.cos(gha)+np.sin(psi)*np.sin(gha)*np.sin(dec),
                 np.sin(psi)*np.cos(dec)])
    Y = np.array([np.sin(psi)*np.sin(gha)-np.cos(psi)*np.cos(gha)*np.sin(dec),
                 np.sin(psi)*np.cos(gha)+np.cos(psi)*np.sin(gha)*np.sin(dec),
                 np.cos(psi)*np.cos(dec)])

    Fplus = np.zeros(ndet)
    Fcross = np.zeros(ndet)
    for i_det in range(ndet):
        Fplus[i_det] = X @ det_response[i_det] @ X - Y @ det_response[i_det] @ Y
        Fcross[i_det] = X @ det_response[i_det] @ Y + Y @ det_response[i_det] @ X

    # (detector)       
    return Fplus, Fcross


def compute_time_delay(params, detstrings):
    ra = params['ra']
    dec = params['dec']
    tgps = params['tgps']

    ndet = len(detstrings)
    i_refdet = reference_detector_index(detstrings)

    det_location = [lal.CachedDetectors[DET_CODE[det_name]].location
                                 for det_name in detstrings]

    gmst = lal.GreenwichMeanSiderealTime(tgps)
    gha = gmst - ra
    esrc = np.array([np.cos(dec)*np.cos(gha), -np.cos(dec)*np.sin(gha), np.sin(dec)])

    time_delay_offset = np.array([-np.dot(esrc, det_location[i_det]) / lal.C_SI for i_det in range(ndet)])
    time_delay_offset_refdet = time_delay_offset[i_refdet]

    # (detector)
    return time_delay_offset - time_delay_offset_refdet


def compute_time_delay_factor(params, f_seq, detstrings):
    tc = params['tc']
    tcoarse = params['tcoarse']

    time_delay = compute_time_delay(params, detstrings)
    time_delay_factor = np.exp(
        -2j * np.pi * f_seq.data
        * (tcoarse + tc + time_delay[:, np.newaxis])
    )
    # (detector, frequency)
    return time_delay_factor


def reference_detector_index(detstrings):
    for name in ["L1", "H1", "V1"]:
        if name in detstrings:
            return detstrings.index(name)
    raise ValueError("Detectors must include H1, L1, or V1")


DET_CODE = {'H1': lal.LHO_4K_DETECTOR,
            'L1': lal.LLO_4K_DETECTOR,
            'V1': lal.VIRGO_DETECTOR}



def compute_strain(params, f_seq, backupNNLO = True, modelist = np.array([[2,1], [2,2], [3,2], [3,3], [4,4]]), uselal = False, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes the strain in the detectors in detstrings for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    modelist: array of modes to use (l, m)
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    if uselal:
        try:
            hplus, hcross = lal_hpc(params, f_seq)
        except:
            hplus, hcross = lal_hpc(add_op(params), f_seq)
    else:        
        hplus, hcross = hpc(params, f_seq, backupNNLO, modelist)

    time_delay_factor = compute_time_delay_factor(params, f_seq, detstrings)
    Fplus, Fcross = compute_response_coeffs(params, detstrings)
    
    # (detector, frequency)
    return ((Fplus[:, np.newaxis] * hplus
             + Fcross[:, np.newaxis] * hcross)
            * time_delay_factor)


def compute_strain_component(params, f_seq, l, mprime, backupNNLO = True, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes the (l, mprime) component of the strain in the detectors in detstrings for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    l: l index
    mprime: mprime index
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    hplus, hcross = hpc_component(params, f_seq, l, mprime, backupNNLO)

    time_delay_factor = compute_time_delay_factor(params, f_seq, detstrings)
    Fplus, Fcross = compute_response_coeffs(params, detstrings)
    
    # Detector strain
    return ((Fplus[:, np.newaxis] * hplus
             + Fcross[:, np.newaxis] * hcross)
            * time_delay_factor)


def compute_strain_components(params, f_seq, backupNNLO = True, modelist = np.array([[2,1], [2,2], [3,2], [3,3], [4,4]]), detstrings = ['H1', 'L1', 'V1']):
    """
    Computes all components of the strain in the detectors in detstrings for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    modelist: array of modes to use (l, m)
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    hpc = np.array([hpc_component(params, f_seq, mode[0], mode[1], backupNNLO) for mode in modelist])

    time_delay_factor = compute_time_delay_factor(params, f_seq, detstrings)
    Fplus, Fcross = compute_response_coeffs(params, detstrings)
    
    # (detector, mode, frequency)
    return ((Fplus[:, np.newaxis, np.newaxis] * hpc[np.newaxis, :, 0, :]
             + Fcross[:, np.newaxis, np.newaxis] * hpc[np.newaxis, :, 1, :])
            *  time_delay_factor[:, np.newaxis, :])


def compute_old_C_prefactor(params, f_seq, l, mprime, backupNNLO = True, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes the prefactor of the (l, mprime) L frame mode of the strain in the detectors in detstrings for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    l: l index
    mprime: mprime index
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    new_C_prefactor = compute_C_prefactor(params, f_seq, l, mprime, backupNNLO, detstrings)

    time_delay_factor = compute_time_delay_factor(params, f_seq, detstrings)
    
    # (detector, frequency)
    return new_C_prefactor * time_delay_factor


def compute_C_prefactor(params, f_seq, l, mprime, backupNNLO = True, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes the prefactor of the (l, mprime) time-dependent L frame mode of the strain in the detectors in detstrings for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    l: l index
    mprime: mprime index
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    thetaJN = params['thetaJN']
    n_freq = len(f_seq.data)
    eulerangles = euler_angles(params, f_seq, mprime, backupNNLO)
    Cplus_J = 1/2*twist_factor_slow(l, mprime, *eulerangles, thetaJN, '+')
    Ccross_J = 1j/2*twist_factor_slow(l, mprime, *eulerangles, thetaJN, 'x')
    zeta = zeta_polarization(params)
    Cplus = np.cos(2*zeta)*Cplus_J + np.sin(2*zeta)*Ccross_J
    Ccross = np.cos(2*zeta)*Ccross_J - np.sin(2*zeta)*Cplus_J

    Fplus, Fcross = compute_response_coeffs(params, detstrings)
    
    # (detector, frequency)
    return (Fplus[:, np.newaxis] * Cplus
             + Fcross[:, np.newaxis] * Ccross)


def compute_td_L_frame_mode(params, f_seq, l, mprime, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes the time-dependent (l, mprime) L frame mode of the strain in the detectors in detstrings for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    l: l index
    mprime: mprime index
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    hL = h_Lframe(params, f_seq, l, -mprime)

    time_delay_factor = compute_time_delay_factor(params, f_seq, detstrings)
    
    # (detector, frequency)
    return hL * time_delay_factor


def compute_C_prefactors(params, f_seq, modelist = np.array([[2,1], [2,2], [3,2], [3,3], [4,4]]), backupNNLO = True, detstrings = ['H1', 'L1', 'V1']):
    """
    Computes all prefactors of the time-dependent L frame modes of the strain in the detectors in detstrings for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    modelist: array of modes to use (l, m)
    backupNNLO: boolean, if True: uses NNLO (next to next to leading order) angles as a backup to MSA (multiple scale analysis) angles if the MSA prescription fails
                         if False: raises an error if the MSA prescription fails
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    thetaJN = params['thetaJN']
    n_freq = len(f_seq.data)
    eulerangles = np.array([euler_angles(params, f_seq, mode[1], backupNNLO) for mode in modelist])
    Cplus_J = np.array([1/2*twist_factor_slow(modelist[i,0], modelist[i,1], *eulerangles[i], thetaJN, '+') for i in range(modelist.shape[0])])
    Ccross_J = np.array([1j/2*twist_factor_slow(modelist[i,0], modelist[i,1], *eulerangles[i], thetaJN, 'x') for i in range(modelist.shape[0])])
    zeta = zeta_polarization(params)
    Cplus = np.cos(2*zeta)*Cplus_J + np.sin(2*zeta)*Ccross_J
    Ccross = np.cos(2*zeta)*Ccross_J - np.sin(2*zeta)*Cplus_J

    Fplus, Fcross = compute_response_coeffs(params, detstrings)
    
    # (detector, mode, frequency)
    return (Fplus[:, np.newaxis, np.newaxis] * Cplus[np.newaxis, :, :]
             + Fcross[:, np.newaxis, np.newaxis] * Ccross[np.newaxis, :, :])


def compute_td_L_frame_modes(params, f_seq, modelist = np.array([[2,1], [2,2], [3,2], [3,3], [4,4]]), detstrings = ['H1', 'L1', 'V1']):
    """
    Computes all time-dependent L frame modes of the strain in the detectors in detstrings for the parameters in the dict params at the frequencies in f_seq.
    params: parameter dictionary
    f_seq: REAL8Sequence LAL object that contains a frequency array
    modelist: array of modes to use (l, m)    
    detstrings: an array of strings specifying the detectors to use in the analysis: H1: LIGO Hanford, L1: LIGO Livingston, V1: Virgo, uses all by default
    """
    hL = np.array([h_Lframe(params, f_seq, mode[0], -mode[1]) for mode in modelist])
    time_delay_factor = compute_time_delay_factor(params, f_seq, detstrings)
    
    # (detector, mode, frequency)
    return hL[np.newaxis, :, :] * time_delay_factor[:, np.newaxis, :]


# Functions for switching between new and old parameters


def add_np(params):
    """
    Computes and adds the new parameters thetaJN and kappa to the parameter dictionary (given a valid parameter dictionary that has inclination (iota) and phiref)
    params: parameter dictionary that has the old angle parameters iota/inclination and phiref
    """
    iota = params['inclination']
    phiref = params['phi_ref']

    # Components in L0 frame
    J = J_from_params(params)

    Nx = np.sin(iota)*np.cos((np.pi/2)-phiref)
    Ny = np.sin(iota)*np.sin((np.pi/2)-phiref)
    Nz = np.cos(iota)
    N = np.array([Nx, Ny, Nz])
    
    params['thetaJN'] = np.arccos(np.dot(J, N)/np.linalg.norm(J))
    
    # Rotate N to J prime frame
    phiJL, thetaJL = phi_theta_from_J(Jx, Jy, Jz)
    N_Jp = rotate_y(rotate_z(N, -phiJL), -thetaJL)
    Nx_Jp = N_Jp[0]
    Ny_Jp = N_Jp[1]
    
    params['kappa'] = np.arctan2(Ny_Jp, Nx_Jp)
    
    return params

def add_op(params):
    """
    Computes and adds the old parameters inclination (iota) and phiref to the parameter dictionary (given a valid parameter dictionary that has the thetaJN and kappa).
    params: parameter dictionary that has the new angle parameters thetaJN and kappa
    """
    thetaJN = params['thetaJN']
    kappa = params['kappa']
    
    # Components in J frame
    Nx = np.sin(thetaJN)
    # Ny = 0
    Nz = np.cos(thetaJN)
    N = np.array([Nx, 0, Nz])
    
    # Rotate to L0 frame
    phiJL = phiJL_func(params)
    thetaJL = thetaJL_func(params)
    NL0 = rotate_z(rotate_y(rotate_z(N, kappa), thetaJL), phiJL)
    
    params['inclination'] = np.arccos(NL0[2])
    params['phi_ref'] = np.pi/2 - np.arctan2(NL0[1], NL0[0])
    
    return params