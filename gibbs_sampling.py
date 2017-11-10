import numpy as np

from mvn import MultivariateNormal

def gibbs_sampling(dist, start, burn_in=100, distance=1, random_seq=False, n=10000):
    '''Gibbs sampling

    Gibbs sampling is useful when one wants to sample from an inaccessible joint distribution,
    but one has access to the full-conditional distributions (i.e the conditional distribution
    of a single variable given all the other variables) and is able to sample from them. 
    This is especially efficient in Bayes-Networks where one can show that the full-conditional 
    boils down to the set of variables in the Markov blanket.

    In Gibbs sampling one establishes a Markov-Chain where at each instant of time a single
    variable is uninstantiated. Then one samples from the full-conditional of this variable
    to determine the variables new instantiation. The process is started by some initial
    sample configuration. 

    It can be shown that Gibbs sampling leads to a stationary distribution (i.e the distribution
    we'd like to sample from), but it is generally unknown when mixin occurs. Many heuristics 
    have been proposed, this code sticks with drawing burn-in samples.

    To reduce the effect of correlation between subsequent draws a number may be specified that
    skips the next-n samples.
    '''
    
    v = np.copy(start)    
    nvars = v.shape[0]    
    mask = np.ones(nvars, dtype=bool)    
    var_ids = np.arange(nvars)

    if random_seq:
        seq_ids = np.random.randint(nvars, size=n*distance+burn_in)
    else:
        seq_ids = np.zeros(n*distance+burn_in, dtype=np.int32)
        seq_ids[1::2] = 1

    idx = 0
    def step():
        nonlocal idx
        i = seq_ids[idx]
        mask[i] = False
        cond = dist.condition_on(var_ids[mask], v[mask])  
        v[i] = cond.sample(1)
        mask[i] = True
        idx += 1
        return v

    def advance(n):
        for _ in range(n):
            step()

    # burn-in
    advance(burn_in)
    
    # sample
    samples = np.empty((n, nvars))
    for i in range(n):
        samples[i] = step()
        advance(distance-1)        

    return samples

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(123)

    true_dist = MultivariateNormal([0., 0.], [[1, 3/5], [3/5, 2]])

    # Gibbs sampling. Although we pass in the true distrubition, Gibbs sampler will only use conditionals.
    gibbs_samples = gibbs_sampling(true_dist, [-3., -3.], burn_in=50, distance=10, n=1000)
   
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x,y = np.meshgrid(x,y)
    xy = np.vstack((x,y)).reshape(2,-1).T
    true_pdf = true_dist(xy)
    plt.contour(x, y, true_pdf.reshape(x.shape))
    
    plt.scatter(gibbs_samples[:, 0], gibbs_samples[:, 1], s=1, label='Gibbs samples')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Gibbs sampling from a multivariate Gaussian distribution')
    plt.savefig('gibbs.png')
    plt.show()