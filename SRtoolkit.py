import numpy as np
import SAPytho.constants as C


def speed(gamma):
    '''v / c = sqrt(1 - 1 / gamma^2)
    '''
    return np.sqrt(1.0 - np.power(gamma, -2))


def speed2(gamma):
    '''(v / c)^2 = 1 - 1 / gamma^2
    '''
    return 1.0 - np.power(gamma, -2)


def LorentzFactor(beta):
    '''Calculating the Lorentz factor:
    gamma = 1 / sqrt(1 - beta^2)
    '''
    return 1.0 / np.sqrt(1.0 - beta**2)


def LorentzFactor2(beta):
    '''Calculating the Lorentz factor:
    gamma^2 = 1 / 1 - beta^2
    '''
    return 1.0 / (1.0 - beta**2)


def momentum(gamma):
    '''Calculating the momentum:
    gamma * beta = sqrt(gamma^2 - 1)
    '''
    return np.sqrt(gamma**2 - 1.0)


def momentum2(gamma):
    '''Calculating the momentum:
    (gamma * beta)^2 = gamma^2 - 1
    '''
    return gamma**2 - 1.0


def gamma_of_p(momentum):
    '''Calculating the Lorentz factor:
    gamma = sqrt(p^2 + 1)
    '''
    return np.sqrt(momentum**2 + 1.0)


def gamma2_of_p(momentum):
    '''Calculating the Lorentz factor:
    gamma^2 = p^2 + 1
    '''
    return momentum**2 + 1.0


def Doppler(gamma, mu):
    '''Doppler factor
    '''
    return 1.0 / (gamma * (1.0 - speed(gamma) * mu))


def nu_obs(nu, z, gamma, muobs):
    '''Compute the observed frequency for a given redshift z.
    '''
    D = Doppler(gamma, muobs)
    return nu * D / (1.0 + z)


def t_obs(t, z, gamma, mu, x=0.0):
    '''From Eq. (2.58) in my thesis.
    '''
    D = Doppler(gamma, mu)
    return (1.0 + z) * ((t / D) + (gamma * x * (speed(gamma) - mu) / C.cLight))


def nu_com(nu, z, gamma, mu):
    '''Compute the comoving frequency for a given redshift z.
    '''
    D = Doppler(gamma, mu)
    return nu * (1.0 + z) / D


def t_com(t, z, gamma, mu, x=0.0):
    '''Time in the comoving frame.
    '''
    D = Doppler(gamma, mu)
    return D * ((t / (1.0 + z)) - (gamma * x * (speed(gamma) - mu) / C.cLight))


def mu_com(mu, gamma):
    '''Viewing angle in the comoving frame.
    '''
    return (mu - speed(gamma)) / (1 - speed(gamma) * mu)


def mu_obs(mu, gamma):
    '''Viewing angle in the observer frame.
    '''
    return (mu + speed(gamma)) / (1 + speed(gamma) * mu)
