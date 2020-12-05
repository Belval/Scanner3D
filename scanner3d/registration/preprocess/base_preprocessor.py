"""
Preprocessor is a step that runs before the registration step
This abstract class defines its implementation
"""

import abc

class BasePreprocessor(abc.ABC):
    def preprocess(self, pcd):
        raise NotImplementedException