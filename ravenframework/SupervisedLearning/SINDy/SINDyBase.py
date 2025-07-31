# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  Created on July 31, 2025

  @author: gonzrl
  Sparse Identification of Nonlinear Dynamical systems (SINDy) base class
"""

#Internal Modules (Lazy Importer)--------------------------------------------------------------------
from ...utils.importerUtils import importModuleLazy
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
# np = importModuleLazy("numpy")
ps = importModuleLazy("pysindy")
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..SupervisedLearning import SupervisedLearning
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# QUESTIONS:
  # - pivotParameterID IS t?
  # - HOW TO HANDLE CustomLibrary? ParameterizedLibrary?
  # - WHERE SHOULD THIS GO: self.model.print(lhs=self.target)
  # - ONLY USE FEATURES WHEN DERIVATIVE INVOLVED?
  # - USE NORMALIZATION
  # - IS IT OKAY TO USE "SINDy" AND "Fourier" IN NAMING, DOES IT COMPLY WITH CAMELBACK?

# NOTES:
  # - USE self.features AND self.target RATHER THAN feature_names INPUT

# TO DO
  # - ADD WARNING FOR MULTIPLE OPTIMIZERS
  # - CREATE DERIVATIVE ESTIMATION CASE
  # - USE HistorySet FOR TIME DEPENDENT CASE
  # - COMPLETE THE REQUIRED FUNCTIONS
  # - ADD OTHER SINDy PARAMETERS
  # - ADD DESCRIPTIONS AND COMMENTS
  # - CHECK WITH DEV FOR CODE FORMATING RULES
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------





class SINDyBase(SupervisedLearning):
  """
    The SINDy (Sparse Identification of Nonlinear Dynamics) algorithm aims to construct a surrogate
    model to identify the governing equations of a dynamical system using sparse regression.
    The surrogate will have the form:
    $\dot{x} = \Theta(X) \Xi$
    where:
      - $x$ is the state variable
      - $\dot{x}$ is the time derivative of the state variable
      - $\Theta(X)$ is the library of candidate functions (e.g., polynomials, trigonometric functions)
      - $\Xi$ is the sparse coefficient matrix, indicating the active terms in the model
  """


  def __init__(self):
    """
      SINDyBase constructor
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
    """

    super().__init__()
    self.printTag                           = 'SINDy'    # Print tag
    self.model                              = None       # the surrogate model itself {'target1':model,'target2':model, etc.}

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = super().getInputSpecification()
    spec.description = r"""Add description"""


    ## SINDy OPTIMIZERS

    def addStlqOptimizer():
      stlqOptimizer = InputData.parameterInputFactory('stlqOptimizer', descr=r"""add desc""")
      stlqOptimizer.addSub(InputData.parameterInputFactory('threshold',contentType=InputTypes.FloatType,
                                                            descr=r"""float, optional (default 0.1)
                                                            Minimum magnitude for a coefficient in the weight vector.
                                                            Coefficients with magnitude below the threshold are set
                                                            to zero."""))
      stlqOptimizer.addSub(InputData.parameterInputFactory('alpha',contentType=InputTypes.FloatType,
                                                            descr=r"""float, optional (default 0.05)
                                                            Optional L2 (ridge) regularization on the weight vector."""))
      stlqOptimizer.addSub(InputData.parameterInputFactory('max_iter',contentType=InputTypes.FloatType,
                                                            descr=r"""int, optional (default 20)
                                                            Maximum iterations of the optimization algorithm."""))
      # stlqOptimizer.addSub(InputData.parameterInputFactory('ridge_kw',contentType=InputTypes.dict,
      #                                                       descr=r"""dict, optional (default None)
      #                                                       Optional keyword arguments to pass to the ridge regression."""))
      stlqOptimizer.addSub(InputData.parameterInputFactory('normalize_columns',contentType=InputTypes.BoolType,
                                                            descr=r"""boolean, optional (default False)
                                                            Normalize the columns of x (the SINDy library terms) before regression
                                                            by dividing by the L2-norm. Note that the 'normalize' option in sklearn
                                                            is deprecated in sklearn versions >= 1.0 and will be removed."""))
      stlqOptimizer.addSub(InputData.parameterInputFactory('copy_X',contentType=InputTypes.BoolType,
                                                            descr=r"""boolean, optional (default True)
                                                            If True, X will be copied; else, it may be overwritten."""))
      # stlqOptimizer.addSub(InputData.parameterInputFactory('initial_guess',contentType=InputTypes.IntegerListType,
      #                                                       descr=r"""np.ndarray, shape (n_features) or (n_targets, n_features),
      #                                                       optional (default None) Initial guess for coefficients ``coef_``.
      #                                                       If None, least-squares is used to obtain an initial guess."""))
      stlqOptimizer.addSub(InputData.parameterInputFactory('verbose',contentType=InputTypes.BoolType,
                                                            descr=r"""bool, optional (default False)
                                                            If True, prints out the different error terms every iteration."""))
      stlqOptimizer.addSub(InputData.parameterInputFactory('sparse_ind',contentType=InputTypes.IntegerListType,
                                                          descr=r"""list, optional (default None)
                                                          Indices to threshold and perform ridge regression upon.
                                                          If None, sparse thresholding and ridge regression is applied to all
                                                          indices."""))
      return stlqOptimizer

    def addSr3Optimizer():
      sr3Optimizer = InputData.parameterInputFactory('sr3Optimizer', descr=r"""add desc""")
      sr3Optimizer.addSub(InputData.parameterInputFactory('threshold',contentType=InputTypes.FloatType,
                                                          descr=r"""float, optional (default 0.1)
                                                          Determines the strength of the regularization. When the
                                                          regularization function R is the L0 norm, the regularization
                                                          is equivalent to performing hard thresholding, and lambda
                                                          is chosen to threshold at the value given by this parameter.
                                                          This is equivalent to choosing lambda = threshold^2 / (2 * nu)."""))
      sr3Optimizer.addSub(InputData.parameterInputFactory('nu',contentType=InputTypes.FloatType,
                                                          descr=r"""float, optional (default 1)
                                                          Determines the level of relaxation. Decreasing nu encourages
                                                          w and v to be close, whereas increasing nu allows the
                                                          regularized coefficients v to be farther from w."""))
      sr3Optimizer.addSub(InputData.parameterInputFactory('tol',contentType=InputTypes.FloatType,
                                                          descr=r"""float, optional (default 1e-5)
                                                          Tolerance used for determining convergence of the optimization
                                                          algorithm."""))
      sr3Optimizer.addSub(InputData.parameterInputFactory('thresholder',contentType=InputTypes.StringType,
                                                          descr=r"""string, optional (default 'L0')
                                                          Regularization function to use. Currently implemented options
                                                          are 'L0' (L0 norm), 'L1' (L1 norm), 'L2' (L2 norm) and 'CAD' (clipped
                                                          absolute deviation). Note by 'L2 norm' we really mean
                                                          the squared L2 norm, i.e. ridge regression"""))
      sr3Optimizer.addSub(InputData.parameterInputFactory('trimming_fraction',contentType=InputTypes.FloatType,
                                                          descr=r"""float, optional (default 0.0)
                                                          Fraction of the data samples to trim during fitting. Should
                                                          be a float between 0.0 and 1.0. If 0.0, trimming is not
                                                          performed."""))
      sr3Optimizer.addSub(InputData.parameterInputFactory('trimming_step_size',contentType=InputTypes.FloatType,
                                                          descr=r"""float, optional (default 1.0)
                                                          Step size to use in the trimming optimization procedure."""))
      sr3Optimizer.addSub(InputData.parameterInputFactory('max_iter',contentType=InputTypes.IntegerType,
                                                          descr=r"""int, optional (default 30)
                                                          Maximum iterations of the optimization algorithm."""))
      # sr3Optimizer.addSub(InputData.parameterInputFactory('initial_guess',contentType=InputTypes.array,
      #                                                     descr=r"""np.ndarray, shape (n_features) or (n_targets, n_features), \
      #                                                     optional (default None)
      #                                                     Initial guess for coefficients ``coef_``.
      #                                                     If None, least-squares is used to obtain an initial guess."""))
      sr3Optimizer.addSub(InputData.parameterInputFactory('normalize_columns',contentType=InputTypes.BoolType,
                                                          descr=r"""boolean, optional (default False)
                                                          Normalize the columns of x (the SINDy library terms) before regression
                                                          by dividing by the L2-norm. Note that the 'normalize' option in sklearn
                                                          is deprecated in sklearn versions >= 1.0 and will be removed."""))
      sr3Optimizer.addSub(InputData.parameterInputFactory('copy_X',contentType=InputTypes.BoolType,
                                                          descr=r"""boolean, optional (default True)
                                                          If True, X will be copied; else, it may be overwritten."""))
      # sr3Optimizer.addSub(InputData.parameterInputFactory('thresholds',contentType=InputTypes.array,
      #                                                     descr=r"""np.ndarray, shape (n_targets, n_features), optional \
      #                                                     (default None)
      #                                                     Array of thresholds for each library function coefficient.
      #                                                     Each row corresponds to a measurement variable and each column
      #                                                     to a function from the feature library.
      #                                                     Recall that SINDy seeks a matrix :math:`\\Xi` such that
      #                                                     :math:`\\dot{X} \\approx \\Theta(X)\\Xi`.
      #                                                     ``thresholds[i, j]`` should specify the threshold to be used for the
      #                                                     (j + 1, i + 1) entry of :math:`\\Xi`. That is to say it should give the
      #                                                     threshold to be used for the (j + 1)st library function in the equation
      #                                                     for the (i + 1)st measurement variable."""))
      sr3Optimizer.addSub(InputData.parameterInputFactory('verbose',contentType=InputTypes.BoolType,
                                                          descr=r"""bool, optional (default False)
                                                          If True, prints out the different error terms every
                                                          max_iter / 10 iterations."""))
      sr3Optimizer.addSub(InputData.parameterInputFactory('unbias',contentType=InputTypes.BoolType,
                                                        descr=r"""bool (default False)
                                                        See base class for definition.  Most options are incompatible
                                                        with unbiasing."""))
      return sr3Optimizer

    spec.addSub(addStlqOptimizer())
    spec.addSub(addSr3Optimizer())

    ## FEATURE LIBRARIES

    def addCustomFeatureLibrary():
      # library_functions: Any,
      # function_names: Any | None = None,
      # interaction_only: bool = True,
      # include_bias: bool = False)
      customFeatureLibrary = InputData.parameterInputFactory('customFeatureLibrary',  descr=r"""add desc""")
      return customFeatureLibrary

    def addFourierFeatureLibrary():
      FourierFeatureLibrary = InputData.parameterInputFactory('FourierFeatureLibrary', descr=r"""add desc""")
      FourierFeatureLibrary.addSub(InputData.parameterInputFactory('n_frequencies',contentType=InputTypes.IntegerType,
                                                  descr=r""" (int, optional (default 1))
                                                  Number of frequencies to include in the library. The library will include functions
                                                  $\sin(x), \sin(2x), \dots, \sin(n_{\text{frequencies}}x)$ for each input feature $x$
                                                  (depending on which of sine and/or cosine features are included)."""))
      FourierFeatureLibrary.addSub(InputData.parameterInputFactory('include_sin',contentType=InputTypes.BoolType,
                                                  descr=r"""(boolean, optional (default True)) – If True, include sine terms in the
                                                  library."""))
      FourierFeatureLibrary.addSub(InputData.parameterInputFactory('include_cos',contentType=InputTypes.BoolType,
                                                descr=r"""(boolean, optional (default True)) – If True, include cosine terms in the
                                                library."""))
      return FourierFeatureLibrary

    def addPolynomialFeatureLibrary():
      polynomialFeatureLibrary = InputData.parameterInputFactory('polynomialFeatureLibrary', descr=r"""add desc""")
      polynomialFeatureLibrary.addSub(InputData.parameterInputFactory('degree',contentType=InputTypes.IntegerType,
                                                  descr=r"""integer, optional (default 2)
                                                  The degree of the polynomial features."""))
      polynomialFeatureLibrary.addSub(InputData.parameterInputFactory('include_interaction',contentType=InputTypes.BoolType,
                                                  descr=r"""boolean, optional (default True)
                                                  Determines whether interaction features are produced.
                                                  If false, features are all of the form ``x[i] ** k``."""))
      polynomialFeatureLibrary.addSub(InputData.parameterInputFactory('interaction_only',contentType=InputTypes.BoolType,
                                                  descr=r"""boolean, optional (default False)
                                                  If true, only interaction features are produced: features that are
                                                  products of at most ``degree`` *distinct* input features (so not
                                                  ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.)."""))
      polynomialFeatureLibrary.addSub(InputData.parameterInputFactory('include_bias',contentType=InputTypes.BoolType,
                                                  descr=r"""boolean, optional (default True)
                                                  If True (default), then include a bias column, the feature in which
                                                  all polynomial powers are zero (i.e. a column of ones - acts as an
                                                  intercept term in a linear model)."""))
      polynomialFeatureLibrary.addSub(InputData.parameterInputFactory('order',contentType=InputTypes.makeEnumType("order", "orderType",["C", "F"]),
                                                  descr=r"""str in {'C', 'F'}, optional (default 'C')
                                                  Order of output array in the dense case. 'F' order is faster to compute,
                                                  but may slow down subsequent estimators."""))
      return polynomialFeatureLibrary

    spec.addSub(addCustomFeatureLibrary())
    spec.addSub(addFourierFeatureLibrary())
    spec.addSub(addPolynomialFeatureLibrary())

    return spec

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """

    # # check if the pivotParameter is among the targetValues
    # if self.pivotParameterID not in self.target:
    #   self.raiseAnError(IOError,"The pivotParameter "+self.pivotParameterID+" must be part of the Target space!")
    super()._handleInput(paramInput)
    # _, notFound = paramInput.findNodesAndExtractValues(['polynomialFeatureLibrary'])
    # # notFound must be empty
    # assert(not notFound)

    featureLibraries = []
    self.optimizer = None
    multipleOptimizerWarningPrinted = False


    featureLibraryMapping = {
      'polynomialFeatureLibrary': ps.PolynomialLibrary,
      'FourierFeatureLibrary': ps.FourierLibrary
    }
    optimizerMapping = {
      'stlqOptimizer': ps.STLSQ,
      'sr3Optimizer': ps.SR3
    }
    for child in paramInput.subparts:
        libraryMap = featureLibraryMapping.get(child.getName())
        optimizerMap = optimizerMapping.get(child.getName())
        if libraryMap is not None:
          args = {cchild.getName(): cchild.value for cchild in child.subparts}
          featureLibraries.append(libraryMap(**args))
        elif optimizerMap is not None:
          args = {cchild.getName(): cchild.value for cchild in child.subparts}
          if self.optimizer is not None and not multipleOptimizerWarningPrinted:
            print("********************* Add warning here *********************")
            multipleOptimizerWarningPrinted = True
          self.optimizer = optimizerMap(**args)

    if not featureLibraries:
      self.featureLibrary = ps.PolynomialLibrary()
    else:
      self.featureLibrary = ps.GeneralizedLibrary(featureLibraries)


    # self.model = ps.SINDy(optimizer=optimizer,
    #                       feature_library=featureLibrary,
    #                       differentiation_method=None, # SINDy differentiation object
    #                       feature_names=self.features, # Really only used in printing.
    #                       t_default=tDefault,
    #                       discrete_time=discreteTime)

  def _train(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape = (n_samples, n_dimensions), an array of input data
      @ In, targetVals, numpy.ndarray, shape = (n_samples, n_timeStep), an array of time series data
    """

    """
    # fit(x: Any,
    # t: Any | None = None,
    # x_dot: Any | None = None,
    # u: Any | None = None,
    # multiple_trajectories: bool = False,
    # unbias: bool = True,
    # quiet: bool = False,
    # ensemble: bool = False,
    # library_ensemble: bool = False,
    # replace: bool = True,
    # n_candidates_to_drop: int = 1,
    # n_subset: Any | None = None,
    # n_models: Any | None = None,
    # ensemble_aggregator: Any | None = None) -> Any

    # x: array-like or list of array-like, shape : n_samples, n_input_features
    # Training data. If training data contains multiple trajectories, x should be a list containing data for each trajectory. Individual trajectories may contain different numbers of samples.

    # t: float, numpy array of shape : n_samples, , or list of numpy arrays, optional (default None)
    # If t is a float, it specifies the timestep between each sample. If array-like, it specifies the time at which each sample was collected. In this case the values in t must be strictly increasing. In the case of multi-trajectory training data, t may also be a list of arrays containing the collection times for each individual trajectory. If None, the default time step t_default will be used.

    # x_dot: array-like or list of array-like, shape : n_samples, n_input_features , optional (default None)
    # Optional pre-computed derivatives of the training data. If not provided, the time derivatives of the training data will be computed using the specified differentiation method. If x_dot is provided, it must match the shape of the training data and these values will be used as the time derivatives.

    # u: array-like or list of array-like, shape : n_samples, n_control_features , optional (default None)
    # Control variables/inputs. Include this variable to use sparse identification for nonlinear dynamical systems for control (SINDYc). If training data contains multiple trajectories (i.e. if x is a list of array-like), then u should be a list containing control variable data for each trajectory. Individual trajectories may contain different numbers of samples.

    # multiple_trajectories: boolean, optional, : default False
    # Whether or not the training data includes multiple trajectories. If True, the training data must be a list of arrays containing data for each trajectory. If False, the training data must be a single array.

    # unbias: boolean, optional : default True
    # Whether to perform an extra step of unregularized linear regression to unbias the coefficients for the identified support. If the optimizer (self.optimizer) applies any type of regularization, that regularization may bias coefficients toward particular values, improving the conditioning of the problem but harming the quality of the fit. Setting unbias==True enables an extra step wherein unregularized linear regression is applied, but only for the coefficients in the support identified by the optimizer. This helps to remove the bias introduced by regularization.

    # quiet: boolean, optional : default False
    # Whether or not to suppress warnings during model fitting.

    # ensemble : boolean, optional (default False)
    # This parameter is used to allow for "ensembling", i.e. the generation of many SINDy models (n_models) by choosing a random temporal subset of the input data (n_subset) for each sparse regression. This often improves robustness because averages (bagging) or medians (bragging) of all the models are usually quite high-performing. The user can also generate "distributions" of many models, and calculate how often certain library terms are included in a model.

    # library_ensemble : boolean, optional (default False)
    # This parameter is used to allow for "library ensembling", i.e. the generation of many SINDy models (n_models) by choosing a random subset of the candidate library terms to truncate. So, n_models are generated by solving n_models sparse regression problems on these "reduced" libraries. Once again, this often improves robustness because averages (bagging) or medians (bragging) of all the models are usually quite high-performing. The user can also generate "distributions" of many models, and calculate how often certain library terms are included in a model.

    # replace : boolean, optional (default True)
    # If ensemble true, whether or not to time sample with replacement.

    # n_candidates_to_drop : int, optional (default 1)
    # Number of candidate terms in the feature library to drop during library ensembling.

    # n_subset : int, optional (default len(time base))
    # Number of time points to use for ensemble

    # n_models : int, optional (default 20)
    # Number of models to generate via ensemble

    # ensemble_aggregator : callable, optional (default numpy.median)
    # Method to aggregate model coefficients across different samples. This method argument is only used if ensemble or library_ensemble is True. The method should take in a list of 2D arrays and return a 2D array of the same shape as the arrays in the list. Example: lambda x: np.median(x, axis=0)
    """
    self.model.fit(x=featureVals)
    # self.model.fit(x=featureVals, x_dot=targetVals)

  def __evaluateLocal__(self,featureVals):
    """
      This method is used to inquire the SINDy model to evaluate (after normalization that in
      this case is not performed) a set of points contained in featureVals.
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """
    return {'y': 0}
    # prediction = self.model.predict(featureVals)
    # prediction_flat = prediction.flatten()
    # return {self.features[i]: prediction_flat[i] for i in range(len(self.features))}


  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def writeXMLPreamble(self, writeTo, targets = None):
    pass

  def writeXML(self, writeTo, targets = None, skip = None):
    pass

  def __confidenceLocal__(self,featureVals):
    pass

  def __resetLocal__(self,featureVals):
    """
      After this method the ROM should be described only by the initial parameter settings
      @ In, featureVals, numpy.ndarray, shape= (n_samples, n_dimensions), an array of input data (training data)
      @ Out, None
    """
    self.amITrained   = False
    self.model        = None
    self.pivotValues  = None
    self.predictError = None
    self.featureVals  = None

  def __returnInitialParametersLocal__(self):
    return {}

  def __returnCurrentSettingLocal__(self):
    return {}
