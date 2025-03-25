"""
This module provides classes for building Feedforward Neural Network (FNN) models using Keras,
custom activation functions, and an Adaptive H-Infinity Filter (AHIF) for estimating the state of a system
based on noisy measurements.

Main Classes:
- FNN: Implements a Feedforward Neural Network model with Keras.
- CustomLeakyReLU: A custom implementation of the Leaky ReLU activation function.
- CustomClippedReLU: A custom implementation of the Clipped ReLU activation function.
- AHIF: An Adaptive H-Infinity Filter for state estimation with noise.
"""

import sys
import logging
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# ------------------------ FNN Class --------------------------------- #
class FNN:
    """
    A class to represent a Feedforward Neural Network (FNN) model.
    
    Attributes:
        input_shape (int): The number of features in the input data.
        output_shape (int): The number of output neurons.
        model (keras.Model, optional): The Keras model instance.
    """

    def __init__(self, input_shape: int, output_shape: int):
        """
        Initializes the FNN with the given input and output shapes.
        """
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        logging.info("FNN instance created with input shape %d and output shape %d.", input_shape, output_shape)

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.output_shape

    def get_model(self):
        """Returns the trained model instance."""
        return self.model

    def build(self):
        """Builds the model architecture."""
        logging.info("Building the model...")
        self.model = keras.Sequential([
            layers.Input(shape=(self.input_shape,)),
            
            layers.Dense(256, activation=keras.activations.relu),
            layers.Dense(256, activation=keras.activations.relu),
            
            layers.Dense(128),               # No activation here
            CustomLeakyReLU(alpha=0.3),      # Apply LeakyReLU as separate layer
            
            layers.Dense(self.output_shape, activation=CustomClippedReLU())
        ])
        logging.info("Model built successfully.")

    def compile(self):
        """Compiles the model with the specified optimizer and loss function."""
        if self.model is None:
            raise ValueError("Model is not built. Call `build()` before `compile()`.")
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,
            decay_rate=0.9,
            staircase=False
        )
        optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
        
        self.model.compile(optimizer=optimizer, loss='mse')
        logging.info("Model compiled successfully.")
        self.model.summary()

# -------------------- Custom Activation Functions ------------------- #

class CustomLeakyReLU(layers.Layer):
    """
    Custom Leaky ReLU activation function.
    """
    def __init__(self, alpha=0.3, **kwargs):
        super(CustomLeakyReLU, self).__init__(**kwargs)
        self.alpha = alpha
        self.leaky_relu = layers.LeakyReLU(alpha=self.alpha)

    def build(self, input_shape):
        super(CustomLeakyReLU, self).build(input_shape)

    def call(self, inputs):
        return self.leaky_relu(inputs)

    def get_config(self):
        config = super(CustomLeakyReLU, self).get_config()
        config.update({"alpha": self.alpha})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomClippedReLU(layers.Layer):
    """
    Custom Clipped ReLU activation function.
    Clips output values to range [0, 1].
    """
    def __init__(self, **kwargs):
        super(CustomClippedReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CustomClippedReLU, self).build(input_shape)

    def call(self, inputs):
        return keras.backend.clip(inputs, 0, 1)

    def get_config(self):
        config = super(CustomClippedReLU, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ------------------- Adaptive H-Infinity Filter (AHIF) ----------------- #

# class AHIF:
#     """
#     Adaptive H-Infinity Filter (AHIF) Implementation.
#     """

#     def __init__(self, process_variance=1e-5, measurement_variance=1e-1, initial_estimate=0, initial_error_covariance=1):
#         """
#         Initializes the AHIF with given parameters.
#         """
#         logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        
#         self._process_variance = process_variance
#         self._measurement_variance = measurement_variance
#         self._estimate = initial_estimate
#         self._error_covariance = initial_error_covariance
        
#         logging.info(f"Initialized AHIF with process_variance={process_variance}, measurement_variance={measurement_variance}, "
#                      f"initial_estimate={initial_estimate}, initial_error_covariance={initial_error_covariance}")

#     def _update(self, measurement: float) -> float:
#         """
#         Performs the prediction and update steps of the filter.
#         """
#         # Prediction
#         predicted_estimate = self._estimate
#         predicted_error_covariance = self._error_covariance + self._process_variance

#         # Kalman Gain Calculation
#         kalman_gain = predicted_error_covariance / (predicted_error_covariance + self._measurement_variance)

#         # Update estimate and error covariance
#         self._estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
#         self._error_covariance = (1 - kalman_gain) * predicted_error_covariance
        
#         return self._estimate

#     def _adapt(self, residuals: list):
#         """
#         Adapts the process variance based on the residuals.
#         """
#         if len(residuals) > 1:
#             residual_std = np.std(residuals)
#             self._process_variance = max(residual_std ** 2, 1e-5)
#         else:
#             self._process_variance = 1e-5

#     def apply(self, data: list) -> np.ndarray:
#         """
#         Applies the AHIF to a series of measurements.
#         """
#         if not data:
#             raise ValueError("Data array is empty")

#         estimates = []
#         residuals = []
        
#         # Initialize estimate with the first data point
#         self._estimate = data[0]
        
#         for measurement in data:
#             estimate = self._update(measurement)
#             estimates.append(estimate)
            
#             # Calculate residuals
#             residual = measurement - estimate
#             residuals.append(residual)
            
#             # Adapt filter every 10 measurements
#             if len(residuals) > 10:
#                 self._adapt(residuals[-10:])
        
#         return np.array(estimates)











class AHIF:
    """
    Adaptive H-Infinity Filter (AHIF) Implementation.
    """

    def __init__(self, process_variance=1e-5, measurement_variance=1e-1, initial_estimate=0, initial_error_covariance=1):
        """
        Initializes the AHIF with given parameters.
        """
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        self._process_variance = process_variance
        self._measurement_variance = measurement_variance
        self._estimate = initial_estimate
        self._error_covariance = initial_error_covariance

        logging.info(f"Initialized AHIF with process_variance={process_variance}, measurement_variance={measurement_variance}, "
                     f"initial_estimate={initial_estimate}, initial_error_covariance={initial_error_covariance}")

    def _update(self, measurement: float) -> float:
        """
        Performs the prediction and update steps of the filter.
        """
        # Prediction Step
        predicted_estimate = self._estimate
        predicted_error_covariance = self._error_covariance + self._process_variance

        # Kalman Gain Calculation
        kalman_gain = predicted_error_covariance / (predicted_error_covariance + self._measurement_variance)

        # Update estimate and error covariance
        self._estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self._error_covariance = (1 - kalman_gain) * predicted_error_covariance

        return self._estimate

    def _adapt(self, residuals: np.ndarray):
        """
        Adapts the process variance based on the residuals.
        """
        if len(residuals) > 1:
            residual_std = np.std(residuals)
            self._process_variance = max(residual_std ** 2, 1e-5)
        else:
            self._process_variance = 1e-5

    def apply(self, data) -> np.ndarray:
        """
        Applies the AHIF to a series of measurements.
        """
        # Ensure data is a NumPy array
        if isinstance(data, list):
            data = np.array(data, dtype=float)
        
        if not isinstance(data, np.ndarray) or data.size == 0:
            raise ValueError("Data array is empty or invalid type")

        estimates = []
        residuals = []

        # Initialize estimate with the first data point
        self._estimate = data[0]

        for measurement in data:
            estimate = self._update(measurement)
            estimates.append(estimate)

            # Calculate residuals
            residual = measurement - estimate
            residuals.append(residual)

            # Adapt filter every 10 measurements
            if len(residuals) >= 10:
                self._adapt(np.array(residuals[-10:]))

        return np.array(estimates)