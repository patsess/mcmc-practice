
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from distcan import Gamma as gamma
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import seaborn as sns

# TODO: add empirical Bayes option (such that frequentist inference is
# TODO used to estimate parameters, and perhaps standard errors, then
# TODO used to specify proposal distributions which could probably be
# TODO fixed)

# TODO: add option to use random standard deviations for proposal dists


class McmcRegressionExample(object):
    def __init__(self, update_params='separately', proposal_dists='adaptive',
                 n_samples=6000, burnin=0.2, thinning=5):
        """Example for regression inference using MCMC

        :param update_params: (str)
            method for updating parameters in Markov chains
        :param proposal_dists: (str)
            method for specifying proposal distributions
        :param n_samples: (int)
            number of samples for Markov chains
        :param burnin: (float)
            proportion of sample of Markov chains to discard as burn-in
        :param thinning: (int)
            keep each {thinning}-th samples of Markov chain for inference
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info('initialising a {} instance'
                         .format(self.__class__.__name__))

        # helpers for properties
        self._dataset = None
        self._priors = None
        self._initial_chain_values = None
        self._initial_proposal_stds = None
        self._target_acceptance_rates = None

        # class helpers
        self._update_params = update_params
        self._proposal_dists = proposal_dists
        self._n_samples = n_samples
        self._burnin = burnin
        self._thinning = thinning

        # initialise attributes
        self._acceptance_tracker = {
            'intercept': {'trace': []},
            'x1_coef': {'trace': []},
            'x2_coef': {'trace': []},
            'residual_std': {'trace': []}
        }
        self.posterior_list = None
        self.posterior_df = None

    @property
    def dataset(self):
        """Random data set for a regression"""
        if self._dataset is None:
            X, y = make_regression(n_samples=1000, n_features=2, noise=10.)
            self._dataset = {'X': X, 'y': y}

        return self._dataset

    @property
    def priors(self):
        """Priors for parameters"""
        if self._priors is None:
            self._priors = {
                'intercept': {
                    'exp': 0.,
                    'std': 1000.,
                    'family': 'gaussian'
                },
                'x1_coef': {
                    'exp': 0.,
                    'std': 1000.,
                    'family': 'gaussian'
                },
                'x2_coef': {
                    'exp': 0.,
                    'std': 1000.,
                    'family': 'gaussian'
                },
                'residual_std': {
                    'exp': 1.,
                    'std': 1000.,
                    'family': 'gamma'
                },
            }

        return self._priors

    @property
    def initial_chain_values(self):
        """Initial values for Markov chains"""
        if self._initial_chain_values is None:
            self._initial_chain_values = {
                'intercept': 0.,
                'x1_coef': 0.,
                'x2_coef': 0.,
                'residual_std': 1.,
            }

        return self._initial_chain_values

    @property
    def initial_proposal_stds(self):
        if self._initial_proposal_stds is None:
            self._initial_proposal_stds = {
                'intercept': 2.,
                'x1_coef': 2.,
                'x2_coef': 2.,
                'residual_std': 2.
            }

        return self._initial_proposal_stds

    @property
    def target_acceptance_rates(self):
        """Note: see https://stats.stackexchange.com/questions/271953/
        acceptance-rate-for-metropolis-hastings-0-5"""
        if self._target_acceptance_rates is None:
            self._target_acceptance_rates = {
                'intercept': 0.25,
                'x1_coef': 0.25,
                'x2_coef': 0.25,
                'residual_std': 0.25
            }
            # TODO: consider increasing targets to (e.g.) 0.8, see
            # TODO https://discourse.pymc.io/t/warning-when-nuts-probability
            # TODO -is-greater-than-acceptance-level/594/3

        return self._target_acceptance_rates

    def run_mcmc(self, verbose=True):
        """Run mcmc to set posterior_df attribute

        :param verbose: (bool) whether to log the mcmc progress
        """
        self.logger.info('running mcmc')

        current_values = self.initial_chain_values.copy()

        if self._update_params == 'together':
            # update all parameters together in the Markov chain
            params_single_update = [list(current_values.keys())]
        elif self._update_params == 'separately':
            # update each parameter separately in the Markov chain
            params_single_update = [[param] for param in current_values.keys()]
        else:
            raise ValueError('unrecognised update_params {}'
                             .format(self._update_params))

        # initialise proposal std parameters in acceptance tracker
        for param, std in self.initial_proposal_stds.items():
            self._acceptance_tracker[param]['std'] = std

        self.posterior_list = []  # initialise
        for i in range(self._n_samples):
            # compute probability of current values (up to proportionality)
            current_logpr = self._get_logpr_upto_proportionality(
                param_values=current_values, priors=self.priors)

            # sample new candidate values for Markov chains
            new_values = self._sample_new_param_values(
                current_values=current_values, iteration=i + 1)

            for params in params_single_update:
                # update relevant parameters
                proposal_values = current_values.copy()
                for param in params:
                    proposal_values[param] = new_values[param]

                # compute probability of proposal values (up to proportionality)
                proposal_logpr = self._get_logpr_upto_proportionality(
                    param_values=proposal_values, priors=self.priors)

                # compute acceptance probability
                acceptance_pr = np.minimum(1., np.exp(
                    proposal_logpr - current_logpr))

                if bool(np.random.binomial(n=1, p=acceptance_pr, size=1)):
                    # update parameter values
                    current_values = proposal_values
                    for param in params:
                        self._acceptance_tracker[param]['trace'] += [1]
                else:
                    for param in params:
                        self._acceptance_tracker[param]['trace'] += [0]

            # update Markov chain
            self.posterior_list += [current_values]
            if verbose and (((i + 1) % 100) == 0):
                self.logger.info('[mcmc progress] done iteration {}'
                                 .format(i + 1))

        self.logger.info('mcmc finished; setting posterior_df attribute')
        self.posterior_df = pd.DataFrame(self.posterior_list)

    def show_frequentist_estimates(self):
        reg = LinearRegression()
        reg.fit(X=self.dataset['X'], y=self.dataset['y'])

        param_names = [param for param in self.priors.keys()
                       if not param.startswith('residual')]
        frequentist_ests = pd.Series(
            [reg.intercept_] + reg.coef_.tolist(), index=param_names)
        self.logger.info('frequentist estimates: {}'
                         .format(frequentist_ests.to_dict()))

    def show_mcmc_results(self):
        # get posterior for inference
        n_samples_burnin = int(np.ceil(self._burnin * len(self.posterior_df)))
        infer_posterior_df = self.posterior_df.iloc[n_samples_burnin:].copy()

        # thin posterior
        thinning_index = np.repeat(range(
            int(np.ceil(len(infer_posterior_df) / float(self._thinning)))),
            repeats=self._thinning)[:len(infer_posterior_df)]
        infer_posterior_df['thinning_index'] = thinning_index
        infer_posterior_df = infer_posterior_df.drop_duplicates(
            'thinning_index', keep='first')
        infer_posterior_df = infer_posterior_df.drop('thinning_index', axis=1)
        # TODO: consider using auto-correlation estimates to thin more smartly

        self.logger.info('mcmc posterior means: {}'
                         .format(infer_posterior_df.mean().to_dict()))
        self.logger.info('mcmc posterior medians: {}'
                         .format(infer_posterior_df.median().to_dict()))

        # plot traces of Markov chains
        for param in self.priors.keys():
            plt.plot(self.posterior_df[param], label=param)
            plt.xlabel('Markov chain iteration')
            plt.ylabel('param value')

        plt.axvline(x=n_samples_burnin, color='grey', linestyle='--')
        plt.legend()
        plt.show()

        # plot marginal posterior probability densities for parameters
        for i, param in enumerate(self.priors.keys()):
            plt.subplot(2, 2, i + 1)
            sns.distplot(infer_posterior_df[param].values, label=param)
            plt.legend()

        plt.show()

    def _sample_new_param_values(self, current_values, iteration,
                                 adaptive_n_req=200):
        """Sample new candidate values for Markov chains"""
        if self._proposal_dists == 'const_stds':
            # use "current" samples as expectations
            proposal_exps = current_values.copy()

            # always accept specified initial proposal stds
            proposal_stds = self.initial_proposal_stds.copy()

        elif self._proposal_dists == 'adaptive':
            # use "current" samples as expectations
            proposal_exps = current_values.copy()

            # adjust "current" proposal stds depending on acceptance rates
            self._set_adaptive_proposal_stds(iteration=iteration,
                                             adaptive_n_req=adaptive_n_req)

            proposal_stds = {
                param: self._acceptance_tracker[param]['std']
                for param in self._acceptance_tracker.keys()}

        elif self._proposal_dists == 'use_traces':
            if iteration >= adaptive_n_req:
                # use Markov chain traces to specify proposal distributions
                post_df = pd.DataFrame(
                    self.posterior_list).iloc[-adaptive_n_req:]

                # use means of post_df for proposal expectations
                proposal_exps = post_df.mean().to_dict()

                # use inflated stds of post_df for proposal stds
                proposal_stds = (post_df.std() * 2.).to_dict()

            else:
                # use "current" samples as expectations
                proposal_exps = current_values.copy()

                # always accept specified initial proposal stds
                proposal_stds = self.initial_proposal_stds.copy()

        else:
            raise ValueError('unrecognised proposal_dists attribute {}'
                             .format(self._proposal_dists))

        # get Gamma parameters for residual_std
        residual_std_shape, residual_std_scale = (
            self.get_shape_and_scale_gamma_params(
                exp=proposal_exps['residual_std'],
                std=proposal_stds['residual_std']))

        new_values = {
            'intercept': norm(proposal_exps['intercept'],
                              scale=proposal_stds['intercept']).rvs(),
            'x1_coef': norm(proposal_exps['x1_coef'],
                            scale=proposal_stds['x1_coef']).rvs(),
            'x2_coef': norm(proposal_exps['x2_coef'],
                            scale=proposal_stds['x2_coef']).rvs(),
            'residual_std': gamma(alpha=residual_std_shape,
                                  beta=residual_std_scale).rvs()
        }
        return new_values

    def _set_adaptive_proposal_stds(self, iteration, adaptive_n_req):
        if (iteration % adaptive_n_req) == 0:
            # look to update proposal stds each adpative_n_req iterations
            for param in self._acceptance_tracker.keys():
                if iteration >= adaptive_n_req:
                    l_ = list(self._acceptance_tracker[param]['trace'])
                    acc_rate = np.mean(l_[-adaptive_n_req:])  # use latest
                    acc_score = acc_rate / float(
                        self.target_acceptance_rates[param])
                    acc_score = np.maximum(0.5, np.minimum(2., acc_score))

                    self._acceptance_tracker[param]['std'] *= acc_score

                    # bound to avoid extreme values
                    initial_std_ = self.initial_proposal_stds[param]
                    assert (initial_std_ >= 0.5)
                    self._acceptance_tracker[param]['std'] = np.maximum(
                        0.5, np.minimum(
                            initial_std_ * 10.,
                            self._acceptance_tracker[param]['std']))
                else:
                    pass  # too few samples; accept initial proposal std

    @staticmethod
    def get_shape_and_scale_gamma_params(exp, std):
        """Get canonical (shape and scale) parameters from expectation and
        standard deviation
        """
        shape_param = (exp ** 2) / float(std ** 2)
        scale_param = (std ** 2) / float(exp)
        return shape_param, scale_param

    def _get_loglikelihood(self, param_values):
        target_params = {
            'exp': (
                    param_values['intercept'] +
                    param_values['x1_coef'] * self.dataset['X'][:, 0] +
                    param_values['x2_coef'] * self.dataset['X'][:, 1]
            ),
            'std': param_values['residual_std']
        }

        log_likelihood = norm(
            target_params['exp'],
            target_params['std']).logpdf(self.dataset['y']).sum()

        return log_likelihood

    def _get_prior_logpr(self, param_values, priors):
        # compute prior probability for each current value
        param_priors = []  # initialise
        for param_name, param_prior in priors.items():
            if param_prior['family'] == 'gaussian':
                param_priors += [
                    norm(param_prior['exp'], param_prior['std']).logpdf(
                        param_values[param_name])]
            elif param_prior['family'] == 'gamma':
                shape_, scale_ = self.get_shape_and_scale_gamma_params(
                    exp=param_prior['exp'], std=param_prior['std'])
                param_priors += [gamma(alpha=shape_, beta=scale_).logpdf(
                    param_values[param_name])]
            else:
                msg = ('family {} not handled for prior probability'
                       .format(param_prior['family']))
                raise NotImplementedError(msg)

        # compute prior probability for current values
        prior_logpr = np.array(param_priors).sum()

        return prior_logpr

    def _get_logpr_upto_proportionality(self, param_values, priors):
        # compute likelihood for parameter values
        loglikelihood = self._get_loglikelihood(param_values=param_values)

        # compute prior probability for parameter values
        prior_logpr = self._get_prior_logpr(param_values=param_values,
                                            priors=priors)

        # compute probability of parameter values (up to proportionality)
        logpr = loglikelihood + prior_logpr
        return logpr


if __name__ == '__main__':
    mcmc = McmcRegressionExample(proposal_dists='use_traces')
    mcmc.run_mcmc()
    mcmc.show_frequentist_estimates()
    mcmc.show_mcmc_results()

    # TODO: compare posteriors (shape etc) with some from pymc3
