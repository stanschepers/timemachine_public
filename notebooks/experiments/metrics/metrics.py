import tensorflow as tf
from keras.utils import metrics_utils
from keras import backend
from keras.losses import mean_absolute_error



class AccuracyAtKYears(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, k=5, min_max_year=None, **kwargs):
        name = f"acc_at_{k}_years"
        if min_max_year is not None:
            min_year, max_year = min_max_year
            k = k / float(max_year - min_year)
            print(f"k used in {name}:", k)
        super(AccuracyAtKYears, self).__init__(accuracy_at_k_years, name=name, k=k, **kwargs)


class CategoricalAccuracyAtKYears(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, k=5, label_years=5, **kwargs):
        name = f"cat_acc_at_{k}_years"
        k = k // label_years
        print(f"k used in in {name}:", k)
        super(CategoricalAccuracyAtKYears, self).__init__(categorical_accuracy_at_k_years,
                                                          name=name, k=k, **kwargs)


class SparseCategoricalAccuracyAtKYears(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, k=5, label_years=5, **kwargs):
        name = f"sparse_cat_acc_at_{k}_years"
        k = k // label_years
        print(f"k used in in {name}:", k)
        super(SparseCategoricalAccuracyAtKYears, self).__init__(categorical_accuracy_at_k_years, sparse=True, name=name,
                                                                k=k, **kwargs)


class MeanAbsoluteErrorForYears(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, min_max_year, **kwargs):
        name = f"mae_for_years"
        min_year, max_year = min_max_year
        super(MeanAbsoluteErrorForYears, self).__init__(mae_for_years, min_year=min_year, max_year=max_year, name=name,
                                                        **kwargs)


def accuracy_at_k_years(y_true, y_pred, k=5, min_year=None, max_year=None):
    [y_pred, y_true], _ = metrics_utils.ragged_assert_compatible_and_get_flat_values([y_pred, y_true])
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    if y_true.dtype != y_pred.dtype:
        y_pred = tf.cast(y_pred, y_true.dtype)
    if min_year is not None:
        k /= (max_year - min_year)
    k = tf.cast(k, y_true.dtype)
    abs_diff = tf.abs(y_true - y_pred)
    acc_at_k = tf.cast(tf.math.greater_equal(k, abs_diff), backend.floatx())
    return tf.cast(acc_at_k, backend.floatx())


def categorical_accuracy_at_k_years(y_true, y_pred, k=5, sparse=False):
    if not sparse:
        y_true = tf.compat.v1.argmax(y_true, axis=-1)
        y_pred = tf.compat.v1.argmax(y_pred, axis=-1)
    k = tf.cast(k, y_true.dtype)
    abs_diff = tf.abs(y_true - y_pred)
    acc_at_k = tf.cast(tf.math.greater_equal(k, abs_diff), backend.floatx())
    return acc_at_k


def mae_for_years(y_true, y_pred, min_year, max_year):
    mae = mean_absolute_error(y_true, y_pred)
    d = tf.cast(max_year - min_year, backend.floatx())
    return tf.cast(mae * d, backend.floatx())