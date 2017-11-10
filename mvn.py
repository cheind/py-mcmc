import numpy as np

class MultivariateNormal:
    '''Multivariate normal distribution'''

    def __init__(self, mu, cov, inv_cov=None):
        self.mu = np.asarray(mu)
        self.cov = np.atleast_2d(cov)
        self.inv_cov = inv_cov or np.linalg.inv(cov)
        self.n = 1. / np.sqrt(2 * np.pi * np.linalg.det(cov))

    def condition_on(self, ids, values):
        '''Condition on the given variables.

        Returns conditional distribution, which is also Gaussian.
        '''
        ids = np.atleast_1d(ids)
        p_mu, p_cov = self.block_form(ids)

        m = len(ids)        
        A = p_cov[:m, :m]
        C = p_cov[:m, m:]
        B = p_cov[m:, m:]

        invA = np.linalg.inv(A)
        mu = p_mu[m:] + C.T @ invA @ (values - p_mu[:m])
        cov = B - C.T @ invA @ C

        return MultivariateNormal(mu, cov)

    def block_form(self, ids):
        '''Rearrange entries of mu and cov so that the given variables take the top-left positions.'''

        # For partitioning we need to rearrange rows and cols into        
        p = np.arange(self.mu.shape[0])
        for i,j in enumerate(ids):
            p[i],p[j] = p[j],p[i]
        
        # Block form. First the ones we condition on then rest.
        return self.mu[p], self.cov[p][:, p]

    def sample(self, n=1):
        return np.squeeze(np.random.multivariate_normal(self.mu, self.cov, size=n))

    def __call__(self, x):
        '''Evaluate the PDF for one or more samples.'''
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        x = np.atleast_2d(x)        
        d = (x - self.mu[None, :])     
        f = lambda x: self.n * np.exp(-0.5 * x.T @ self.inv_cov @ x)       
        return np.squeeze(np.apply_along_axis(f, 1, d))

    @staticmethod
    def estimate(x):
        '''Estimate distribution parameters from samples.'''

        mu = np.mean(x, axis=0)
        cov = np.cov(x.T)

        return MultivariateNormal(mu, cov)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mvn = MultivariateNormal([0., 0.], [[1, 3/5], [3/5, 2]])    

    x = np.arange(-4, 4, 0.1)
    plt.plot(x, mvn.condition_on(0, 0.0)(x))
    plt.plot(x, mvn.condition_on(1, 4.0)(x))
    plt.plot(x, mvn.condition_on(1, -4.0)(x))
    plt.show()
