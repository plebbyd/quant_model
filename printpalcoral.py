import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from typing import Union, Optional
import numpy as np

model_file = 'model_quant_8_full_edgetpu.tflite'



def _pairwise_distances_no_broadcast_helper(X, Y):
    """Internal function for calculating the distance with numba. Do not use.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        First input samples

    Y : array of shape (n_samples, n_features)
        Second input samples

    Returns
    -------
    distance : array of shape (n_samples,)
        Intermediate results. Do not use.

    """
    if len(X.shape) <=1:
        X = X.reshape(-1, 1)
    if len(Y.shape) <=1:
        Y = Y.reshape(-1,1)
    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
            raise ValueError("pairwise_distances_no_broadcast function receive"
                             "matrix with different shapes {0} and {1}".format(
                X.shape, Y.shape))
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()



class AutoEncoder:

    def __init__(self,
            model_file : str,
            threshold : float = 0.5
        ) -> None:
        self.interpreter = edgetpu.make_interpreter(model_file)
        self.interpreter.allocate_tensors()
        self.threshold = threshold

    def _set_input(self, X : np.ndarray) -> None:
        common.input_tensor(self.interpreter)[:,] = X

    def _get_input_size(self) -> tuple:
        return self.interpreter.get_input_details()[0]['shape']

    def infer(self, X : np.ndarray, scaler : bool, normalize : bool = False) -> np.ndarray:
        # Set the input first
        if normalize:
            if not np.any((X > 1.0)|(X < 0.0)):
                print("[warning] - All values in X are between 0.0 and 1.0 - ou may be normalizing the data twice")
                print("[warning] - Ignore this message if you know what you are doing.")
            X = scaler.transform(X)
        self._set_input(X)
        self.interpreter.invoke()
        return common.output_tensor(self.interpreter, 0).copy()

    def to_dequantized_array(self, X : np.ndarray) -> np.ndarray:
        '''
        X is the output_tensor() returned from self.infer()
        '''
        X = X.flatten()
        output_details = self.interpreter.get_output_details()[0]
        if np.issubdtype(output_details['dtype'], np.integer):
            scale, zero_point = output_details['quantization']
            # Always convert to np.int64 to avoid overflow on subtraction.
            return scale * (X.astype(np.int64) - zero_point)
        return X.copy()


    def classify(self, X : np.ndarray, scaler : bool, normalize : bool = False, threshold : Union[float, None] = None) -> np.ndarray:
        raw_scores = self.infer(X, normalize=normalize, scaler=scaler)
        dequantized = self.to_dequantized_array(raw_scores)
        distances = _pairwise_distances_no_broadcast_helper(X, dequantized)

        threshold = threshold if threshold is not None else self.threshold
        classes = (distances > threshold).astype('int').ravel()
        return classes

    def perf_analyze(self, batch_size : int = 1, iterations : int = 100):
        row = np.random.rand(*self._get_input_size())
        start = time.perf_counter()
        self._set_input(row)
        for i in range(iterations):
            self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        print(f'Model performance summary: {iterations} iterations | {inference_time/iterations} second/inference')
        return inference_time / iterations
