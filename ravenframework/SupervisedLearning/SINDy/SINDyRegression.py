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

  def _handleInput(self, paramInput): # WHY DO I NEED THIS FUNCTION?
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)

  def _train(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape = (n_samples, n_dimensions), an array of input data
      @ In, targetVals, numpy.ndarray, shape = (n_samples, n_timeStep), an array of time series data
    """

    self.model.fit(x=featureVals, x_dot=targetVals) # x_dot is y

    # print("******************************************************************************")
    # self.model.print(lhs=self.target) # WHERE SHOULD THIS GO?
    # print("******************************************************************************")


  def __evaluateLocal__(self,featureVals):
    """
      This method is used to inquire the SINDy model to evaluate (after normalization that in
      this case is not performed) a set of points contained in featureVals.
      @ In, featureVals, numpy.ndarray, shape= (n_requests, n_dimensions), an array of input data
      @ Out, returnEvaluation , dict, dictionary of values for each target (and pivot parameter)
    """

    # IN WHAT CASE IS n_requests > 1? THIS WILL NOT WORK
    prediction = self.model.predict(featureVals)
    prediction_flat = prediction.flatten()
    return {self.target[i]: prediction_flat[i] for i in range(len(self.target))}
