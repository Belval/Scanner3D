"""
A preprocessor sequence is a wrapper around multiple preprocessing steps
"""


class PreprocessorSequence:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def preprocess(self, pcds):
        was_list = True
        if not isinstance(pcds, list):
            was_list = False
            pcds = [pcds]

        new_pcds = []
        for pcd in pcds:
            for preprocessor in self.preprocessors:
                pcd = preprocessor.preprocess(pcd)
            new_pcds.append(pcd)

        if was_list:
            return new_pcds
        else:
            return new_pcds[0]
