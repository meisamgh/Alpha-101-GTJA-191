import numpy as np

from quant_research.models.train import ModelSpec, build_model, fit_fold


def test_scaler_is_fit_on_training_only():
    x_train = np.array([[0.0], [2.0]])
    y_train = np.array([0.0, 1.0])
    x_validation = np.array([[10_000.0]])
    model = fit_fold(build_model(ModelSpec("ridge")), x_train, y_train, x_validation)
    assert model.named_steps["scaler"].mean_[0] == 1.0
