import math

import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

A1 = 1./3.
A2 = 1./3.
A3 = 1./3.
MU = .2

class HMM:
    # This class converted with modifications from http://kastnerkyle.github.io/blog/2014/05/22/single-speaker-speech-recognition/
    # Implementation of: http://articles.ircam.fr/textes/Bevilacqua09b/index.pdf

    def __init__(self, n_states, reference):
        self.n_states = n_states
        self.reference = reference
        #self.random_state = np.random.RandomState(0)
        
        # Initial state
        # left-to-right HMM, we start with state 1
        self.prior = np.zeros(self.n_states)
        self.prior[0] = 1.

        self.A = np.zeros((self.n_states, self.n_states))
        #self.A = self._stochasticize(self.random_state.rand(self.n_states, self.n_states))
        for i in range(self.n_states):
            self.A[i, i] = A1
            if (i+1) < self.A.shape[1]:
                self.A[i, i+1] = A2
            if (i+2) < self.A.shape[1]:
                self.A[i, i+2] = A3
        self.A[-1, -1] = 1.
        
        self.mu = np.array([MU]*len(self.reference))
           
    def _forward(self, B):
        log_likelihood = 0.
        T = B.shape[1]
        alpha = np.zeros(B.shape)
        #T = B.shape[1]
        #T = B.shape[0]
        #print(B)
        #alpha = np.zeros((self.n_states, self.n_states, self.reference.shape[1]))
        #for t in range(self.n_states):
        for t in range(T):
            if t == 0:
                #print(B[:, t].shape)
                #print(self.prior.ravel().shape)
                #alpha[t] = (B.transpose(1,0) * self.prior).transpose(1,0)
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                #alpha[t] = B * np.dot(self.A.T, alpha[t-1])
                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])

            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            #log_likelihood = log_likelihood + np.log(alpha_sum)
            log_likelihood = log_likelihood + alpha_sum

        #print(B[:, 3])
        return log_likelihood, alpha
    
    def _state_likelihood(self, obs):
        obs = np.atleast_2d(obs)
        B = np.zeros((self.n_states, obs.shape[0]))
        for s in range(self.n_states):
            #B[s, :] = st.multivariate_normal.pdf(obs.T, mean=self.mu)
            b = np.zeros(obs.shape[0])
            for o in range(obs.shape[0]):
                b[o] = 0.
                b[o] = (1./(self.mu[s]*math.sqrt(2*math.pi))) * \
                    math.exp(
                        -( (obs[o][0]-self.reference[s][0])**2 / (2*(self.mu[s]**2)) )
                    )
            #B[s, :] = self._normalize(b)
            B[s, :] = b

            #Needs scipy 0.14
            #B[s, :] = st.multivariate_normal.pdf(obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T)

            #This function can (and will!) return values >> 1
            #See the discussion here for the equivalent matlab function
            #https://groups.google.com/forum/#!topic/comp.soft-sys.matlab/YksWK0T74Ak
            #Key line: "Probabilities have to be less than 1,
            #Densities can be anything, even infinite (at individual points)."
            #This is evaluating the density at individual points...
        return B
    
    def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)
    
    def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)

if __name__ == "__main__":
    reference_signal = np.concatenate((
        np.zeros(50),
        np.sin(np.linspace(-np.pi, np.pi, 40)),
        np.zeros(50),
        np.sin(np.linspace(-np.pi, np.pi, 40)),
        np.zeros(50)))

    noise = np.random.normal(0,.1,230)
    offset = .2
    test_signal = np.concatenate((
        np.random.normal(0,.1,70) + offset,
        noise + reference_signal + reference_signal + offset))

    #test_signal = np.concatenate((
    #    np.zeros((50,)),
    #    reference_signal))

    # test signal 2 is just noise
    test_signal2 = np.random.normal(0,1,230)

    # plt.plot(reference_signal)
    # plt.plot(test_signal)
    # plt.plot(test_signal2)
    # plt.show()

    r = np.reshape(reference_signal, (-1, 1))
    t = np.reshape(test_signal[:150], (-1, 1))
    t2 = np.reshape(test_signal2, (-1, 1))

    # Build HMM based on reference data

    h = HMM(len(r), r)
    B = h._state_likelihood(t)
    B2 = h._state_likelihood(t2)
    B3 = h._state_likelihood(r)
    lik, alpha = h._forward(B)
    for t in range(alpha.shape[0]):
        print(np.argmax(alpha[t, :]))
    print("Likelihood for test data: {}".format(lik))
    print("Likelihood for noise data: {}".format(h._forward(B2)[0]))
    print("Likelihood for reference data: {}".format(h._forward(B3)[0]))
