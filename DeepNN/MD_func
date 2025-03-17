import numpy as np
from math import sqrt

def MD(relative_st_vec, state_sigmas, state_covariances):
    state_variances = np.square(state_sigmas)
    covariance_matrix = np.array([[state_variances[0], state_covariances[0],state_covariances[1]],
                                 [state_covariances[0], state_variances[1], state_covariances[2]],
                                 [state_covariances[1], state_covariances[2], state_variances[2]]])
    covariance_matrix_inv = np.linalg.inv(covariance_matrix)
    MD = sqrt(np.dot(np.dot(relative_st_vec, covariance_matrix_inv), relative_st_vec))
    return MD