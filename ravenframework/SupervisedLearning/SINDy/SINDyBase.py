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
np = importModuleLazy("numpy")
ps = importModuleLazy("pysindy")
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..SupervisedLearning import SupervisedLearning
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------


#########################################################################################################################

#########################################################################################################################


class SINDyBase(SupervisedLearning):

  def __init__(self):
    super().__init__()
    # handling time series?
    self._dynamicHandling = True
    # # initial settings for the ROM (coming from input) # FROM DMDBASE NOT SURE IF NEEDED
    # self.settings = {}
    # SINDy-based model parameters (used in the initialization of the SINDy models)
    self.SINDyParams = {}
    # SINDy model
    self.model = None

    # target indeces (positions in self.target list)
    self.targetIndices = None
    # This flag is needed because the SINDy based model has an issue with single target (space dimension == 1)
    self.singleTarget = False



  @classmethod
  def getInputSpecification(cls):
    specs = super().getInputSpecification()

    ## SINDy OPTIMIZERS

    ## ADD EnsembleOptimizer

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

    specs.addSub(addStlqOptimizer())
    specs.addSub(addSr3Optimizer())

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

    specs.addSub(addCustomFeatureLibrary())
    specs.addSub(addFourierFeatureLibrary())
    specs.addSub(addPolynomialFeatureLibrary())

    return specs

  def _handleInput(self, paramInput):
    super()._handleInput(paramInput)

    featureLibraries = []
    optimizer = None
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
          if optimizer is not None and not multipleOptimizerWarningPrinted:
            self.raiseADebug(f"Two optimizers provided, but only one can be used. Using {optimizer}") ## PROPER WAY SHOW USER A WARNING?
            multipleOptimizerWarningPrinted = True
          optimizer = optimizerMap(**args)

    if not featureLibraries:
      featureLibrary = ps.PolynomialLibrary()
    else:
      featureLibrary = ps.GeneralizedLibrary(featureLibraries)

    self.SINDyParams['optimizer'] = optimizer
    self.SINDyParams['featureLibrary'] = featureLibrary


  def initializeModel(self, SINDyParams):

    self.SINDyParams = SINDyParams

    self.model = ps.SINDy(optimizer=self.SINDyParams['optimizer'],
                  feature_library=self.SINDyParams['featureLibrary'],
                  differentiation_method=self.SINDyParams['differentiationMethod'], # SINDy differentiation object
                  # feature_names=self.features, # CAUSING ISSUE, ONLY GOOD FOR PRINTING MODEL
                  t_default=self.SINDyParams['tDefault'])

######
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
    """
      Specific local method for printing anything desired to xml file at the begin of the print.
      @ In, writeTo, xmlUtils.StaticXmlElement instance, element to write to
      @ In, targets, list, list of targets for whom information should be written.
      @ Out, None
    """
    # add description
    super().writeXMLPreamble(writeTo, targets)
    description  = ' This XML file contains the main information of the SINDy-based ROM .'
    description += ''
    writeTo.addScalar('ROM',"description",description)

  def writeXML(self, writeTo, targets = None, skip = None):
    """
      Adds requested entries to XML node.
      @ In, writeTo, xmlTuils.StaticXmlElement, element to write to
      @ In, targets, list, optional, list of targets for whom information should be written
      @ In, skip, list, optional, list of targets to skip
      @ Out, None
    """
    if not self.amITrained:
      self.raiseAnError(RuntimeError,'ROM is not yet trained!')
    if skip is None:
      skip = []


        # check what
    # FROM DMDBASE

    # what = ['features','timeScale','eigs','amplitudes','modes','dmdTimeScale'] + list(self.dmdParams.keys())
    # if targets is None:
    #   readWhat = what
    # else:
    #   readWhat = targets
    # for s in skip:
    #   if s in readWhat:
    #     readWhat.remove(s)
    # if not set(readWhat) <= set(what):
    #   self.raiseAnError(IOError, "The following variables specified in <what> node are not recognized: "+ ",".join(np.setdiff1d(readWhat, what).tolist()) )
    # else:
    #   what = readWhat

    # target = self.name
    # toAdd = list(self.dmdParams.keys())

    # for add in toAdd:
    #   if add in what :
    #     writeTo.addScalar(target,add,self.dmdParams[add])
    # targNode = writeTo._findTarget(writeTo.getRoot(), target)
    # if "features" in what:
    #   writeTo.addScalar(target,"features",' '.join(self.features))
    # if "timeScale" in what:
    #   writeTo.addScalar(target,"timeScale",' '.join(['%.6e' % elm for elm in self.pivotValues.ravel()]))
    # if "dmdTimeScale" in what:
    #   writeTo.addScalar(target,"dmdTimeScale",' '.join(['%.6e' % elm for elm in self._getTimeScale()]))
    # if "eigs" in what:
    #   eigsReal = " ".join(['%.6e' % self.model._reference_dmd.eigs[indx].real for indx in
    #                    range(len(self.model._reference_dmd.eigs))])
    #   writeTo.addScalar("eigs","real", eigsReal, root=targNode)
    #   eigsImag = " ".join(['%.6e' % self.model._reference_dmd.eigs.imag[indx] for indx in
    #                            range(len(self.model._reference_dmd.eigs))])
    #   writeTo.addScalar("eigs","imaginary", eigsImag, root=targNode)
    # if "amplitudes" in what and 'amplitudes' in dir(self.model._reference_dmd) and self.model._reference_dmd.amplitudes is not None:
    #   ampsReal = " ".join(['%.6e' % self.model._reference_dmd.amplitudes.real[indx] for indx in
    #                    range(len(self.model._reference_dmd.amplitudes))])
    #   writeTo.addScalar("amplitudes","real", ampsReal, root=targNode)
    #   ampsImag = " ".join(['%.6e' % self.model._reference_dmd.amplitudes.imag[indx] for indx in
    #                            range(len(self.model._reference_dmd.amplitudes))])
    #   writeTo.addScalar("amplitudes","imaginary", ampsImag, root=targNode)
    # if "modes" in what:
    #   nSamples = self.featureVals.shape[0]
    #   delays = max(1, int(self.model._reference_dmd.modes.shape[0] / nSamples))
    #   loopCnt = 0
    #   noSampled = False
    #   if nSamples * delays !=  self.model._reference_dmd.modes.shape[0]:
    #     nSamples = self.model._reference_dmd.modes.shape[0]
    #     noSampled = True
    #   for smp in range(nSamples):
    #     valDict = {'real':'', 'imaginary': ''}
    #     for _ in range(delays):
    #       valDict['real'] += ' '.join([ '%.6e' % elm for elm in self.model._reference_dmd.modes[loopCnt,:].real]) + ' '
    #       valDict['imaginary'] += ' '.join([ '%.6e' % elm for elm in self.model._reference_dmd.modes[loopCnt,:].imag]) +' '
    #       loopCnt += 1
    #     if noSampled:
    #       attributeDict = {"index":f'{loopCnt}'}
    #     else:
    #       attributeDict = {self.features[index]:'%.6e' % self.featureVals[smp,index] for index in range(len(self.features))}
    #     if delays > 1:
    #       attributeDict['shape'] = f"({self.model._reference_dmd.modes.shape[1]},{delays})"
    #     writeTo.addVector("modes","realization" if not noSampled else "element",valDict, root=targNode, attrs=attributeDict)



  def __confidenceLocal__(self,featureVals):
    """
      The confidence associate with a set of requested evaluations
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, None
    """
    pass

  def __resetLocal__(self,featureVals):
    """
      After this method the ROM should be described only by the initial parameter settings
      @ In, featureVals, numpy.ndarray, shape= (n_samples, n_dimensions), an array of input data (training data)
      @ Out, None
    """
    self.amITrained   = False
    self.model = {}
    self.pivotValues  = None
    self.featureVals  = None

  def __returnInitialParametersLocal__(self):
    """
      This method returns the initial parameters of the SM
      @ In, None
      @ Out, params, dict, the dict of the SM settings
    """
    return self.SINDyParams

  def __returnCurrentSettingLocal__(self):
    """
      This method is used to pass the set of parameters of the ROM that can change during simulation
      @ In, None
      @ Out, params, dict, the dict of the SM settings
    """
    return self.SINDyParams

