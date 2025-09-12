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
  Sparse Identification of Nonlinear Dynamical systems ROM Creation
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
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class SINDy(SINDyBase):

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(SINDy, cls).getInputSpecification()

    specs.addSub(InputData.parameterInputFactory('pivotParameter',contentType=InputTypes.StringType,
                                                descr=r"""defines the pivot variable (e.g., time) that represents the
                                                independent monotonic variable""", default='time'))



    ## ADD DIFFERENTIATION PARAMETERS

    return specs


  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """

    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['pivotParameter'])
    # notFound must be empty
    assert(not notFound)
    self.pivotParameterID  = settings.get("pivotParameter")  # pivot parameter
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError,f"The pivotParameter {self.pivotParameterID} must be part of the Target space!")
    if len(self.target) < 2:
      self.raiseAnError(IOError,f"At least one Target in addition to the pivotParameter {self.pivotParameterID} must be part of the Target space!")

    self.targetIndices = tuple([i for i,x in enumerate(self.target) if x != self.pivotID])

    # add parameters to SINDyParams set here
    self.SINDyParams['differentiationMethod'] = None # DEFAULT
    self.SINDyParams['tDefault'] = 1  # DEFAULT

    self.initializeModel(self.SINDyParams)


  def _train(self,featureVals,targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, numpy.ndarray, shape = (n_samples, n_dimensions), an array of input data
      @ In, targetVals, numpy.ndarray, shape = (n_samples, n_timeStep), an array of time series data
    """

    pivotParamIndex   = self.target.index(self.pivotParameterID)
    self.pivotValues  = targetVals[0,:,pivotParamIndex]

    n = len(targetVals[0]) # IS THIS ALREADY AVAILABLE?
    targetValsList = [targetVals[i][:,self.targetIndices] for i in range(targetVals.shape[0])]
    featureValsList = [np.repeat([fv], n, axis=0) for fv in featureVals]
    featureValsList = [np.column_stack((fv, self.pivotValues.reshape(n, 1))) for fv in featureValsList]

    self.model.fit(x=featureValsList, x_dot=targetValsList, t=self.pivotValues, multiple_trajectories=True) # t not really used here because x_dot provided?

  def __evaluateLocal__(self,featureVals):

    n = len(self.pivotValues)
    featureValsRepeated = np.repeat(featureVals, n, axis=0)
    featureValsWithTime = np.column_stack((featureValsRepeated, self.pivotValues.reshape(n, 1)))
    result = self.model.predict(featureValsWithTime)

    returnEvaluation = {self.pivotParameterID:self.pivotValues}

    for i, index in enumerate(self.targetIndices):
      target = self.target[index]
      returnEvaluation[target] = np.array(result[:, i])

    print("**********************************************************************************************")
    print("featureVals:\n", featureVals)
    print("returnEvaluation:\n", returnEvaluation)
    print("**********************************************************************************************")
    return returnEvaluation
