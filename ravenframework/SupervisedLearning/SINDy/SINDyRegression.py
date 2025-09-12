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
  Sparse regression via PySINDy libraries
"""

#Internal Modules (Lazy Importer)--------------------------------------------------------------------
from ...utils.importerUtils import importModuleLazy
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
ps = importModuleLazy("pysindy")
np = importModuleLazy("numpy")
#External Modules End--------------------------------------------------------------------------------


#Internal Modules------------------------------------------------------------------------------------
from ...SupervisedLearning.SINDy import SINDyBase
#Internal Modules End--------------------------------------------------------------------------------

class SINDyRegression(SINDyBase):
  """
    This surrogate model performs sparse regression using the PySINDy library.

    The usual form of the problem SINDy solves:
    $\dot{x} = \Theta(X) \Xi$
    where:
      - $x$ is the state variable
      - $\dot{x}$ is the time derivative of the state variable
      - $\Theta(X)$ is the library of candidate functions (e.g., polynomials, trigonometric functions)
      - $\Xi$ is the sparse coefficient matrix, indicating the active terms in the model

    In this special case:
    $y = \Theta(X) \Xi$
    where:
      - $x$ is the independent variable
      - $y$ is the dependent variable
      - $\Theta(X)$ is the library of candidate functions (e.g., polynomials, trigonometric functions)
      - $\Xi$ is the sparse coefficient matrix, indicating the active terms in the model
  """
  def __init__(self):
    """
      SINDyRegression constructor
      @ In, kwargs, dict, an arbitrary dictionary of keywords and values
    """
    super().__init__()
    self.uniqueVals = None # flag to indicate targets only have a single unique value

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)

    self.SINDyParams['differentiationMethod'] = None # DEFAULT
    self.SINDyParams['tDefault'] = 1  # DEFAULT

    self.initializeModel(self.SINDyParams)


  def _train(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape = (n_samples, n_dimensions), an array of input data
      @ In, targetVals, numpy.ndarray, shape = (n_samples, n_timeStep), an array of time series data
    """
    # check if all targets only have a single unique value, just store that value, no need to fit/train
    if all([len(np.unique(targetVals[:,index])) == 1 for index in range(targetVals.shape[1])]):
      self.uniqueVals = [np.unique(targetVals[:,index])[0] for index in range(targetVals.shape[1]) ]
    else:
      # the multi-target is handled by the internal wrapper
      self.uniqueVals = None
      self.model.fit(x=featureVals, x_dot=targetVals) # x_dot is y

  def __evaluateLocal__(self,featureVals):
    """
      Evaluates a point.
      @ In, featureVals, np.array, list of values at which to evaluate the ROM
      @ Out, returnDict, dict, dict of all the target results
    """
    if self.uniqueVals is not None:
      outcomes =  self.uniqueVals
    else:
      outcomes = self.model.predict(featureVals)

    outcomes = np.atleast_1d(outcomes)
    #possibilities for predict results are:
    # (n_samples,) or (n_samples, n_targets)
    if len(outcomes.shape) == 1 and len(self.target) == 1:
      returnDict = {self.target[0]:outcomes}
    elif len(outcomes.shape) == 1:
      #this might only be possible for scikitlearn bugs
      returnDict = {key:value for (key,value) in zip(self.target,outcomes)}
    else:
      returnDict = {key: outcomes[:, i] for i, key in enumerate(self.target)}

    return returnDict
