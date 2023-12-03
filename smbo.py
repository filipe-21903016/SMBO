import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
import numpy as np
import typing
import warnings
warnings.filterwarnings("ignore")
from warnings import catch_warnings
from warnings import simplefilter
from scipy.stats import norm

class SequentialModelBasedOptimization(object):

    def __init__(self, random_state = 1):
        """
        Initializes the Gaussian Process model with empty variables and sets the default values for the list of runs (capital R),
        the incumbent (theta_inc being the best found hyperparameters, theta_inc_performance being the performance associated with it),
        and additional variables for tracking the best Gaussian process and its scores.

        Parameters
        ----------
        random_state : int, default=1
            The random seed for the Gaussian Process model.

        The Gaussian Process model is initialized with a Matern kernel and the specified random state.

        Returns
        -------
        None
        """ 
        self.model = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=sklearn.gaussian_process.kernels.Matern(), random_state=random_state)
        self.capital_r: typing.List[typing.Tuple[np.array, float]] = None
        self.theta_inc_performance: float = None
        self.theta_inc: np.ndarray = None

    def initialize(self, capital_phi: typing.List[typing.Tuple[np.array, float]]) -> None:
        """
        Initializes the model with a set of initial configurations and their performances.

        This method sets the initial configurations and their performances, typically accuracy, to initialize the model
        before it can make recommendations on which configurations are in good regions. Note that higher performance values
        are preferred.

        Parameters
        ----------
        capital_phi : List[Tuple[np.array, float]]
            A list of tuples, where each tuple contains a configuration and its performance (typically, accuracy).

        Returns
        -------
        None
        """
        self.capital_r = capital_phi
        for configuration, performance in capital_phi:
            if self.theta_inc_performance is None or performance > self.theta_inc_performance:
                self.theta_inc = configuration
                self.theta_inc_performance = performance

    def fit_model(self) -> None:
        """
        Fits the Gaussian Process model to the provided configurations and their corresponding performances.

        This method uses the configurations and performances stored in the `capital_r` attribute to train the Gaussian Process model.

        Returns
        -------
        None
        """
        configurations = [theta[0] for theta in self.capital_r]
        performances = [theta[1] for theta in self.capital_r]
        with catch_warnings():
            simplefilter("ignore")
        self.model.fit(configurations, performances)

    def select_configuration(self, capital_theta: np.array) -> np.array:
        """
        Determines the expected improvement (EI) of each configuration based on the internal Gaussian Process model.
        Note that higher values of EI indicate better configurations.

        Parameters
        ----------
        capital_theta : array-like, shape (n, m)
            An array where each column represents a hyperparameter and each row represents a configuration.

        Returns
        -------
        EI : array-like, shape (n,)
            An array of the same size as the number of configurations, where each element represents the EI of a given configuration.
        """
        ei = self.expected_improvement(self.model, self.theta_inc_performance, capital_theta)
        return capital_theta[np.argmax(ei)]

    @staticmethod
    def expected_improvement(model: sklearn.gaussian_process.GaussianProcessRegressor,
                             f_star: float, theta: np.array) -> np.array:
        """
        Computes the Expected Improvement (EI) acquisition function for a given set of configurations.

        The EI acquisition function determines which configurations are good and which are not good based on the internal
        Gaussian Process model. Higher values of EI indicate better configurations.

        Parameters
        ----------
        model : sklearn.gaussian_process.GaussianProcessRegressor
            The internal Gaussian Process model (should be fitted already).
        f_star : float
            The current incumbent (theta_inc).
        theta : np.array
            A (n, m) array, where each column represents a hyperparameter and each row represents a configuration.

        Returns
        -------
        np.array
            A size n vector, where each element represents the EI of a given configuration.
        """
        mu_values, sigma_values = model.predict(theta, return_std=True)
        ei_values = np.array([])
        for i in range(len(mu_values)):
            mu = mu_values[i]
            sigma = sigma_values[i]
            z = (mu - f_star)/sigma
            ei_values = np.append(ei_values, (mu - f_star) * norm.cdf(z) + sigma * norm.pdf(z))
        
        return ei_values

    def update_runs(self, run: typing.Tuple[np.array, float]):
        """
        Updates the list of runs by appending the given run and updating the best Gaussian process values if necessary.

        The method adds the provided run to the `capital_r` list and updates the `theta_inc`, `theta_inc_performance`, `best_gaussian`, and `best_gaussian_scores` variables if the current run's performance is better than the previous best.

        Parameters
        ----------
        run : typing.Tuple[np.array, float]
            A tuple containing a configuration (hyperparameters) and its performance.
        """
        self.capital_r.append(run)
        configuration = run[0]
        performance = run[1]
        if performance > self.theta_inc_performance:
            self.theta_inc = configuration
            self.theta_inc_performance = performance
        return None
    
    def return_best_configuration(self):
        return self.theta_inc_performance, self.theta_inc