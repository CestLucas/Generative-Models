"""
Template for Question 2 of hwk3.
@author: Samuel Lavoie
"""
import torch
import q2_sampler
import q2_model


def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    # LP regularized cost function: E_nu(f(y)) - E_mu(f(x)) + theta E[max(0, l2(grad(x_hat))-1)^2]
    # x_hat = tx + (1-t)y, t in U(0,1)
    batch_size = x.shape[0]
    sampler = iter(q2_sampler.distribution1(0, batch_size))
    sample = next(sampler)[:,1]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = torch.Tensor(sample)[:,None].to(device)
    
    x_hat = t * x + (1 - t) * y
    x_hat = x_hat.to(device)
    x_hat.requires_grad = True
    
    f = critic(x_hat)
    grad = torch.autograd.grad(outputs=f, inputs=x_hat,
                              grad_outputs=torch.ones(f.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    
    return (torch.max(torch.zeros(batch_size, 1), (grad.norm(p=2, dim=1) - 1)) ** 2).mean()


def vf_wasserstein_distance(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    # wasserstein objective function: max E_p[f(x)] - E_q[f(x)]
    return torch.mean(critic(x)) - torch.mean(critic(y))


def vf_squared_hellinger(x, y, critic):
    """
    Complete me. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Nowazin et al: https://arxiv.org/pdf/1606.00709.pdf
    In other word, x are samples from the distribution P and y are samples from the distribution Q. Please note that the Critic is unbounded. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Squared Hellinger.
    :return: (FloatTensor) - shape: (1,) - Estimate of the Squared Hellinger
    """
    # objective Function (saddle)
    # F(theta, w) = E_p[gf(V_w(x))] - E_q[-f*(gf(V_w(x)))]
    # minimize wrt theta and maximize wrt w
    
    # squared hellinger output activation: 1 - exp(-v)
    # squared hellinger conjugate: t / (1 - t)
    sq_hellinger_x = 1 - torch.exp(-critic(x))
    sq_hellinger_y = 1 - torch.exp(-critic(y))
    conjugate_y = sq_hellinger_y / (1 - sq_hellinger_y)
    return torch.mean(sq_hellinger_x) - torch.mean(conjugate_y)


if __name__ == '__main__':
    # Example of usage of the code provided for answering Q2.5 as well as recommended hyper parameters.
    model = q2_model.Critic(2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler1 = iter(q2_sampler.distribution1(0, 512))
    theta = 0
    sampler2 = iter(q2_sampler.distribution1(theta, 512))
    lambda_reg_lp = 50 # Recommended hyper parameters for the lipschitz regularizer.
