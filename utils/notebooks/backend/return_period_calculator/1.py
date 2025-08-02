import logging
import os
import pathlib
import shutil
from typing import Optional, Mapping

import numpy as np
import pandas as pd
import xarray
from matplotlib import pyplot as plt
from tqdm import tqdm
import theoretical_distribution_utilities as tdu
import empirical_distribution_utilities as edu
import grubbs_beck_tester as gbt
from notebooks.backend import data_paths
from notebooks.backend.evaluation_utils import RETURN_PERIODS
from notebooks.backend.return_period_calculator import base_fitter, exceptions, extract_peaks_utilities, \
    plotting_utilities
from notebooks.backend.return_period_calculator.generalized_expected_moments_algorithm import \
    _central_moments_from_data_and_parameters, _central_moments_from_data, _check_convergence_norm1, _MAX_ITERATIONS
from notebooks.backend.return_period_metrics import calculate_return_period_performance_metrics

KN_TABLE_FILENAME = r'C:\Users\Administrator\Downloads\b17\1\KNtable.csv'


def _load_kn_table(
        file: str = KN_TABLE_FILENAME
) -> Mapping[int, float]:
    kn_table_series = pd.read_csv(file, index_col='0')
    kn_table_series.index = kn_table_series.index.astype(int)
    return kn_table_series.to_dict()['1']


def _ema(
        systematic_record: np.ndarray,
        pilf_threshold: float,
        convergence_threshold: Optional[float] = 1e-8,
) -> dict[str, float]:
    """Implements the Genearlized Expected Moments Algorithm (EMA).

  This is the full fitting procedure from Bulletin 17c, and is the main
  difference between that and the 1981 USGS protocol from Bulletin 17b.
  This algorithm is described on page 27 of Bulletin 17c, with full
  implementation details given in Appendix 7 (page 82).

  Args:
    systematic_record: Systematic data record of flood peaks.
      Must be in transformed units.
    pilf_threshold: As determined by the MGBT test. Units must match
      systematic record.
    convergence_threshold: Convergence threshold to be applied to the first
      norm of the moments of the EMA-estimated distribution. The default value
      is 1e-10.

  Returns:
    dict of fit parameters keyed by parameter name according to Equations 8-10.

  Raises:
    NumericalFittingError if there are nans or infs in iterative algorithm.
  """

    # Turn all data into lower and upper bounds.
    num_pilfs = len(systematic_record[systematic_record < pilf_threshold])
    lower_bounds = systematic_record[systematic_record >= pilf_threshold]
    upper_bounds = systematic_record[systematic_record >= pilf_threshold]
    if num_pilfs > 0:
        lower_bounds = np.concatenate([
            np.full(num_pilfs, -np.inf),
            lower_bounds,
        ])
        upper_bounds = np.concatenate([
            np.full(num_pilfs, pilf_threshold),
            upper_bounds,
        ])

    # Steps in this algorithm are listed on pages 83-84.
    # Step #1a: Initial estimates of central moments.
    # These are used for the first expected value calculations and also for the
    # first convergence check.
    previous_moments = _central_moments_from_data(data=systematic_record)

    # Step #2: Expectation-maximization loop.
    converged = False
    iteration = 0
    while not converged:
        iteration += 1

        # Update distribution parameters.
        parameters = _parameters_from_central_moments(moments=previous_moments)

        # Step #2a: Update expected moments.
        expected_moments = _central_moments_from_data_and_parameters(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            parameters=parameters,
        )

        # Error checking.
        if (np.isnan([val for val in expected_moments.values()]).any()
                or np.isinf([val for val in expected_moments.values()]).any()):
            raise exceptions.NumericalFittingError(
                routine='GEMA',
                condition=f'NaN or inf found on iteration {iteration}.',
            )

        # Step #2b: Weight with regional skew.
        # TODO(gsnearing) Implement regional skew.

        # Step #2c: Check for convergence.
        converged = _check_convergence_norm1(
            current_moments=expected_moments,
            previous_moments=previous_moments,
            convergence_threshold=convergence_threshold,
        )
        previous_moments = expected_moments

        if iteration > _MAX_ITERATIONS:
            raise exceptions.NumericalFittingError(
                routine='GEMA',
                condition='max iterations reached'
            )

    return parameters


def _parameters_from_central_moments(
        moments: dict[str, float],
) -> dict[str, float]:
    """Estimate distribution parameters from central moments."""
    # This is equation 7-13 (page 83).
    alpha, beta, tau = tdu.parameters_from_moments(
        mean=moments['M'],
        std=moments['S'],
        skew=moments['G'],
    )
    return {'alpha': alpha, 'beta': beta, 'tau': tau}


class GEMAFitter(base_fitter.BaseFitter):
    """Estimates return periods using the Generalized Expected Moments Algorithm.

  This is the baseline algorithm from Bulletin 17c.
  """

    def __init__(
            self,
            data: np.ndarray,
            kn_table: Optional[Mapping[int, float]] = None,
            convergence_threshold: Optional[float] = None,
            log_transform: bool = True
    ):
        """Constructor for a GEMA distribution fitter.

    Fits parameters of a log-Pearson-III distribution with the iterative EMA
    procedure, using the generalized versions of the distribution moments
    and interval moments.

    Args:
      data: Flow peaks to fit in physical units.
      kn_table: Custom test statistics table to override the Kn Table from
        Bulletin 17b in the Standard Grubbs Beck Test (GBT).
      convergence_threshold: Convergence threshold to be applied to the first
        norm of the moments of the EMA-estimated distribution. The default value
        is 1e-10.
      log_transform: Whether to transform the data before fitting a
        distribution.
    """
        super().__init__(
            data=data,
            log_transform=log_transform
        )

        # Find the PILF threshold.
        # TODO(gsnearing): Use Multiple Grubbs-Beck Test instead of Grubbs-Beck
        # Test.
        self._pilf_tester = gbt.GrubbsBeckTester(
            data=self.transformed_sample,
            kn_table=kn_table,
        )

        # Run the EMA algorithm.
        self._distribution_parameters = _ema(
            systematic_record=self.transformed_sample,
            pilf_threshold=self._pilf_tester.pilf_threshold,
            convergence_threshold=convergence_threshold,
        )

    @property
    def type_name(self) -> str:
        return self.__class__.__name__

    def exceedance_probabilities_from_flow_values(
            self,
            flows: np.ndarray,
    ) -> np.ndarray:
        """Predicts exceedance probabilities from streamflow values.

    Args:
      flows: Streamflow values in physical units.

    Returns:
      Predicted exceedance probabilities.

    Raises:
      ValueError if return periods are requested for zero-flows.
    """
        if np.any(flows <= 0):
            raise ValueError('All flow values must be positive.')
        transformed_flows = self._transform_data(flows)
        return 1 - tdu.pearson3_cdf(
            alpha=self._distribution_parameters['alpha'],
            beta=self._distribution_parameters['beta'],
            tau=self._distribution_parameters['tau'],
            values=transformed_flows,
        )

    def flow_values_from_exceedance_probabilities(
            self,
            exceedance_probabilities: np.ndarray,
    ) -> np.ndarray:
        """Predicts from pre-fit log-linear regression.

    Args:
      exceedance_probabilities: Probability of exceeding a particular flow value
        in a given year.

    Returns:
      Flow values corresponding to requeseted exceedance_probabilities.

    Raises:
      ValueError if cumulative probailities are outside realistic ranges, or
        include 0 or 1.
    """
        transformed_flow_values = tdu.pearson3_invcdf(
            alpha=self._distribution_parameters['alpha'],
            beta=self._distribution_parameters['beta'],
            tau=self._distribution_parameters['tau'],
            quantiles=(1 - exceedance_probabilities),
        )
        return self._untransform_data(data=transformed_flow_values)


_DEFAULT_PLOTTING_RETURN_PERIODS = np.array([1.01, 2, 5, 10, 20, 50, 100])
# The minimum years of record must be 2 or larger, since 2 are necessary to
# fit a linear trend.
_MIN_YEARS_OF_RECORD = 5


class ReturnPeriodCalculator():
    """Primary object for calculating return periods."""

    def __init__(
            self,
            peaks_series: Optional[pd.Series] = None,
            hydrograph_series: Optional[pd.Series] = None,
            hydrograph_series_frequency: Optional[pd.Timedelta] = None,
            is_stage: bool = False,
            extract_peaks_function=extract_peaks_utilities.extract_annual_maximums,
            use_simple_fitting: bool = False,
            use_log_trend_fitting: bool = False,
            kn_table: Optional[Mapping[int, float]] = None,
            verbose: bool = True,
    ):
        """Constructor for Return Period Calculator.

    Args:
      peaks_series: Option to allow users to supply their own peaks, instead
        of using utilities to extract peaks from a hydrograph.
      hydrograph_series: Systematic data record as a Pandas series in physical
        units (e.g., cms, m3/s) with dates as indexes. Peaks will be extracted
        from this hydrograph unless provided by `peaks_series`.
      hydrograph_series_frequency: Frequency of timestep in the hydrgraph
        series. Must be supplied if peaks_series is not supplied.
      is_stage: Indicates whether the hydrograph and/or peaks data are stage
        (as opposed to discharge). If stage data are used, they are not
        log-transformed.
      extract_peaks_function: Function to find "floods" to use for fitting a
        return period distribution. The default is annual maximums, and this
        should not be changed unless you know what you are doing and why.
      use_simple_fitting: Use simple distribution fitting instead of the
        Expected Moments Algorithm (EMA). This does not account for Potentially
        Impactful Low Floods (PILFs), zero-flows, or historical flood data.
      use_log_trend_fitting: Use log-linear regression on empirical plotting
        positions to fit a return period estimator, instead of fitting a
        distribution.
      kn_table: Custom test statistics table to override the Kn Table from
        Bulletin 17b in the Standard Grubbs Beck Test (GBT).
      verbose: Whether to print status messages during runtime.

    Raises:
      ValeError if neither hydrograph nor peaks are provided.
    """
        # Extract peaks or work with peaks supplied by the user.
        if hydrograph_series is not None and peaks_series is None:
            self._hydrograph = hydrograph_series
            if hydrograph_series_frequency is None:
                raise ValueError('User must supply the time frequency of the '
                                 'hydrograph series.')
            self._peaks = extract_peaks_function(
                hydrograph_series,
                hydrograph_series_frequency,
            )
        elif hydrograph_series is not None and peaks_series is not None:
            self._hydrograph = hydrograph_series
            self._peaks = peaks_series.dropna()
        elif hydrograph_series is None and peaks_series is not None:
            self._hydrograph = peaks_series
            self._peaks = peaks_series.dropna()
        else:
            raise ValueError('Must supply either a hydrograph series or a peaks '
                             'series.')

        if len(self._peaks) < _MIN_YEARS_OF_RECORD:
            raise exceptions.NotEnoughDataError(
                num_data_points=len(self._peaks),
                data_requirement=_MIN_YEARS_OF_RECORD,
                routine='Return Period Calculator',
            )

        # If working with stage data, don't log-transform.
        self.is_stage = is_stage

        # TODO(gsnearing): Implement record extension with nearby sites (Appdx. 8).

        # Fit the distribution. First, we try using the Generalized Expected Moments
        # Algorithm, which is the standard approach in Bulletin 17c. If that fails
        # (e.g., sample size is too small), we revert to log-log linear regression
        # against simple empircal plotting positions. The user can request simple
        # distribution fitting instead of GEMA, and this also reverts to regression
        # if the sample size is too small or if there is any other type of numerical
        # error.
        run_backup_fitter = False
        try:
            if use_log_trend_fitting:
                raise exceptions.AskedForBackupError(method='Log-Liner Regression')

            if use_simple_fitting:
                self._fitter = tdu.SimpleLogPearson3Fitter(
                    data=self._peaks.values,
                    log_transform=(not self.is_stage)
                )
            else:
                self._fitter = GEMAFitter(
                    data=self._peaks.values,
                    kn_table=kn_table,
                    log_transform=(not self.is_stage)
                )
        except (exceptions.NumericalFittingError, exceptions.NotEnoughDataError):
            if verbose:
                logging.exception('Reverting to using the regression fitter as a '
                                  'backup')
            run_backup_fitter = True
        except exceptions.AskedForBackupError:
            run_backup_fitter = True

        if run_backup_fitter:
            self._fitter = edu.LogLogTrendFitter(
                data=self._peaks.values,
                log_transform=(not self.is_stage)
            )

    def __len__(self) -> int:
        return len(self._peaks)

    @property
    def fitter_type(self) -> str:
        return self._fitter.type_name

    def plot_hydrograph_with_peaks(self):
        """Plot the hydrograph with values used for return period analysis."""
        plotting_utilities.plot_hydrograph_with_peaks(
            hydrograph_series=self._hydrograph,
            peaks_series=self._peaks,
        )
        plt.show()

    # TODO(gsnearing): Make this show zeros, PILFs, historical.
    def plot_fitted_distribution(self):
        """Plot the empirical and theoretical (fit) floods distributions."""
        plotting_utilities.plot_fitted_distribution(
            data=self._peaks,
            fitter=self._fitter,
        )
        plt.show()

    def plot_exceedence_probability_distribution(self):
        """Plot the exceedence probability distribution."""
        plotting_utilities.plot_exceedence_probability_distribution(
            fitter=self._fitter,
        )
        plt.show()

    def plot_hydrograph_with_return_periods(
            self,
            return_periods: Optional[np.ndarray] = None,
    ):
        """Plot hydrograph with overlaid return periods."""
        if return_periods is None:
            return_periods = _DEFAULT_PLOTTING_RETURN_PERIODS
        return_period_values = self.flow_values_from_return_periods(
            return_periods=return_periods)
        plotting_utilities.plot_hydrograph_with_return_periods(
            hydrograph_series=self._hydrograph,
            return_period_values={rp: val for rp, val in zip(
                return_periods, return_period_values)},
        )
        plt.show()

    def flow_values_from_return_periods(
            self,
            return_periods: np.ndarray,
    ) -> np.ndarray:
        # TODO(gsnearing): Also return confidence intervals.
        """Flow values for an array of return periods.

    Args:
      return_periods: Return periods for which to calculate flow values.

    Returns:
      Estimated flow values in physical units for given return periods.
    """
        return self._fitter.flow_values_from_return_periods(
            return_periods=return_periods)

    def flow_value_from_return_period(
            self,
            return_period: float,
    ) -> float:
        """Flow value for a single return period.

    Args:
      return_period: Return period for which to calculate flow value.

    Returns:
      Estimated flow value in physical units for a given return period.
    """
        return self._fitter.flow_values_from_return_periods(
            return_periods=np.array([return_period]))[0]

    def flow_values_from_percentiles(
            self,
            percentiles: np.ndarray,
    ) -> np.ndarray:
        # TODO(gsnearing): Also return confidence intervals.
        """Flow values for an array of distribution percentiles.

    Args:
      percentiles: CDF percentiles for which to calculate flow values.

    Returns:
      Estimated flow values in physical units for given return periods.
    """
        return self._fitter.flow_values_from_exceedance_probabilities(
            exceedance_probabilities=1 - percentiles)

    def percentiles_from_flow_values(
            self,
            flows: np.ndarray,
    ) -> np.ndarray:
        # TODO(gsnearing): Also return confidence intervals.
        """CDF percentiles for a given set of flow values.

    Args:
      flows: flow values in physical units for given return periods

    Returns:
      Estimated CDF percentiles for which to calculate flow values.
    """
        return self._fitter.exceedance_probabilities_from_flow_values(
            flows=flows)

    def return_periods_from_flow_values(
            self,
            flows: np.ndarray,
    ) -> np.ndarray:
        # TODO(gsnearing): Also return confidence intervals.
        """Return period for a given flow value.

    Args:
        flows: Flow values for which to calculate a return period in physical
          units.

    Returns:
      Estimated return period for given flow values.
    """
        mask = np.where(flows > 0)
        return_periods = np.full_like(flows, np.nan, dtype=float)
        return_periods[mask] = self._fitter.return_periods_from_flow_values(
            flows=flows[mask])
        return return_periods


if __name__ == '__main__':
    # datain = pd.read_csv(r'C:\Users\Administrator\Downloads\b17\1\ex1.csv')
    # datain = datain.values[:, 1]
    kn_table = _load_kn_table()

    hydro_series = pd.read_csv(r'E:\cyh\global_streamflow_model_paper-main\global_streamflow_model_paper-main\Model\result\old\116095.csv',
                               index_col=0)
    obs_series = hydro_series.iloc[:,0]
    # obs_series.index = pd.to_datetime(obs_series.index)
    sim_seres = hydro_series.iloc[:,1]
    # obs_series.index = pd.to_datetime(obs_series.index)

    temporal_resolution = pd.Timedelta("1D")
    # sim_flow_values_ = fitter.flow_values_from_return_periods()
    # calculator = ReturnPeriodCalculator(hydrograph_series=hydro_series,hydrograph_series_frequency=temporal_resolution,kn_table=kn_table,
    #                                     is_stage=False)
    a = calculate_return_period_performance_metrics(observations=obs_series,predictions=sim_seres,temporal_resolution="1D")
    # print()