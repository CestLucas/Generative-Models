"""
Template for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch


def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    # log_likelihood_bernoulli
    # L = log (mu^tar * (1-mu)^(1-t)) = tar*log(mu) + (1-tar)log(1-mu)
    return torch.sum(target * torch.log(mu) + (1-target) * torch.log(1-mu), dim=1)


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    # log normal
    # L = log (1/(sd*sqrt(2*pi))exp(-1/2((x-mu)/sd)^2)) = -log(sd*sqrt(2*pi)) -1/2((x-mu)/sd)^2)
    # L = -log(sqrt(2*pi*var) - 1/2((x-mu)^2/var))
    var = torch.exp(logvar)
    return torch.sum(-torch.log((2*np.pi*var)**0.5) - (z-mu)**2/(2*var), dim=1)


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    # log_mean_exp = log(1/k* sum^k(exp(y_i^k-a_i))) + a_i
    y_max, idx = torch.max(y, dim=1)
    y_max = y_max.reshape(-1,1) # reshape: (batch_size, )
    return torch.log(torch.mean(torch.exp(y - y_max), dim=1, keepdim=True)) + y_max 


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    # kld
    # ref: https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions
    # D_KL[q||p] = 1/2(trace(var_p ^{-1} var_q) + (mu_p - mu_q)^T var_p ^{-1} (mu_p - mu_q) - input_size + log(det_p/det_q))
    diff_mu = mu_p - mu_q
    
    trace = torch.sum(torch.exp(logvar_q - logvar_p), dim=1)
    diff_inv_diff = torch.sum(diff_mu / torch.exp(logvar_p) * diff_mu, dim=1)
    input_size = mu_q.size(1)
    log = torch.sum(logvar_p, dim=1) - torch.sum(logvar_q, dim=1) # shape: (batch_size, 1)
    
    return (trace + diff_inv_diff - input_size + log) / 2


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    # kld
    # define the distributions
    q = torch.distributions.MultivariateNormal(mu_q, torch.diag_embed(torch.exp(logvar_q)))
    p = torch.distributions.MultivariateNormal(mu_p, torch.diag_embed(torch.exp(logvar_p)))
    z = q.sample()
    
    return torch.mean(q.log_prob(z) - p.log_prob(z), dim=1)