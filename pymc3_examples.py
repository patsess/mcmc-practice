
import logging
import matplotlib.pyplot as plt
import pymc3 as pm
from sklearn.datasets import make_regression
import seaborn as sns

# note: also see https://towardsdatascience.com/
# markov-chain-monte-carlo-in-python-44f7e609be98


class Pymc3RegressionExample(object):
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info('initialising a {} instance'
                         .format(self.__class__.__name__))

        # helpers for properties
        self._dataset = None

    @property
    def dataset(self):
        """Random data set for a regression"""
        if self._dataset is None:
            X, y = make_regression(n_samples=1000, n_features=2, noise=10.)
            self._dataset = {'X': X, 'y': y}

        return self._dataset

    def run_mcmc(self, spec_method='flexible'):
        with pm.Model() as mdl:
            if spec_method == 'flexible':
                # specify priors
                self.logger.info('specifying priors')
                intercept = pm.Normal('intercept', mu=0., sd=1000.)
                x1_coef = pm.Normal('x1_coef', mu=0., sd=1000.)
                x2_coef = pm.Normal('x2_coef', mu=0., sd=1000.)
                # residual_std = pm.HalfCauchy('sigma', beta=10, testval=1.)
                residual_std = pm.Gamma('residual_std', mu=1., sd=1000.,
                                        testval=1.)

                # specify likelihood
                self.logger.info('specifying likelihood')
                mu = (intercept +
                      x1_coef * self.dataset['X'][:, 0] +
                      x2_coef * self.dataset['X'][:, 1])
                likelihood = pm.Normal(
                    'y', mu=mu, sd=residual_std, observed=self.dataset['y'])

            elif spec_method == 'patsy_glm':
                data_dict = {
                    'y': self.dataset['y'],
                    'x1': self.dataset['X'][:, 0],
                    'x2': self.dataset['X'][:, 1],
                }

                self.logger.info('specifying model using patsy glm method')
                pm.glm.GLM.from_formula('y ~ x1 + x2', data_dict)

            else:
                raise ValueError('unrecognised spec_method {}'
                                 .format(spec_method))

            # run mcmc (using automatically chosen sampler, e.g. NUTS sampling)
            self.logger.info('running mcmc')
            trace = pm.sample(6000, njobs=1, tune=1000)
            # note: 'tune' argument handles the burn-in

            # show results (with no thinning)
            n_burnin_samples = 0  # burn-in handled above
            msg = ('summary of marginal posteriors (no thinning):\n{}'
                   .format(pm.summary(trace, start=n_burnin_samples).round(2)))
            self.logger.info(msg)
            pm.traceplot(trace, skip_first=n_burnin_samples)
            plt.show()

            self._show_custom_plots(
                trace=trace,
                params=['intercept', 'x1_coef', 'x2_coef', 'residual_std'],
                burnin=n_burnin_samples)

    @staticmethod
    def _show_custom_plots(trace, params, burnin=0, thinning=5):
        trace_dict = {
            param: trace.get_values(param, burn=burnin, thin=thinning)
            for param in params}

        # plot traces of Markov chains
        for param in params:
            plt.plot(trace_dict[param], label=param)
            plt.xlabel('Markov chain iteration')
            plt.ylabel('param value')

        plt.legend()
        plt.show()

        # plot marginal posterior probability densities for parameters
        assert (len(params) == 4)
        for i, param in enumerate(params):
            plt.subplot(2, 2, i + 1)
            sns.distplot(trace_dict[param], label=param)
            plt.legend()

        plt.show()


if __name__ == '__main__':
    reg = Pymc3RegressionExample()
    reg.run_mcmc(spec_method='flexible')
