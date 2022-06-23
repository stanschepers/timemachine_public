import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd

import coral_ordinal as coral

from .utils import *
from .metrics import *


class TaskMixin:
    task_type = None
    n_classes = 1

    def _get_top(self):
        raise NotImplemented()

    def _compile_model(self, model, learning_rate, period):
        raise NotImplemented()


class FeatureExtractionMixin:
    def _get_feature_extraction(self):
        raise NotImplemented()


class BaseExperiment:
    def __init__(self, n_classes=1, min_year=None, max_year=None,
                 name=None, period=5, config=None):
        self.name = name
        self.n_classes = n_classes
        self.min_year = min_year
        self.max_year = max_year
        self.period = period
        self.callback_dir = get_path(name)
        self.input_shape = (224, 224, 3)
        self.config = config if config is not None else {}

    def _resolve_name(self, given_name):
        return f"{given_name}_{self.task_type}"

    def get_model(self):
        model = tf.keras.Sequential([
            self._get_data_augementation(),
            self._get_feature_extraction(),
            self._get_top(),
        ], name=self.name)
        
        return self._build(model)

    def pretrain(self, model, train_data, val_data, epochs=10):
        pretrain_lr = self._get_pretrain_learning_rate()
        self._compile_model(model, learning_rate=pretrain_lr, period=self.period)

        pretrain_history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=get_callbacks(self.callback_dir),
        )

        return model, pretrain_history

    def finetune(self, model, train_data, val_data, epochs=10):
        model.get_layer(name="feature_extraction").trainable = True

        finetune_lr = self._get_finetune_learning_rate()
        self._compile_model(model, learning_rate=finetune_lr, period=self.period)

        finetune_history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=get_callbacks(self.callback_dir),
        )

        return model, finetune_history

    def _get_data_augementation(self):
        return get_data_augmentation()

    def _get_pretrain_learning_rate(self) -> float:
        return 1e-3

    def _get_finetune_learning_rate(self) -> float:
        return 1e-5
    
    def _build(self, model):
        model.build([None, 224, 224, 3])
        return model

    def evaluate(self, model, test_data):
        return model.evaluate(test_data)

    def run(self, train_data, val_data, test_data, pretrain_epochs=50, finetune_epochs=50):
        performance = []
        
        try:
            print(f"[{self.name}][experiment] start")
            print(f"[{self.name}][model creation] callback directory: {self.callback_dir}")
            os.makedirs(self.callback_dir, exist_ok=True)

            print(f"[{self.name}][model creation] start")
            model = self.get_model()
            print(f"[{self.name}][model creation] end")

            print(f"[{self.name}][pretrain] start")
            pretrain_history = self.pretrain(model, train_data, val_data, epochs=pretrain_epochs)
            print(f"[{self.name}][pretrain] end")

            pretrain_val_results = self.evaluate(model, val_data)
            print(f"[{self.name}][pretrain][evaluation] validation dataset: {pretrain_val_results}")
            performance.append(("pretrain", "val", str(pretrain_val_results)))


            pretrain_test_results = self.evaluate(model, test_data)
            print(f"[{self.name}][pretrain][evaluation] test dataset: {pretrain_test_results}")
            performance.append(("pretrain", "test", str(pretrain_test_results)))

            print(f"[{self.name}][finetune] start")
            finetune_history = self.finetune(model, train_data, val_data, epochs=finetune_epochs)
            print(f"[{self.name}][finetune] end")

            finetune_val_results = self.evaluate(model, val_data)
            print(f"[{self.name}][finetune][evaluation] validation dataset: {finetune_val_results}")
            performance.append(("finetune", "val", str(finetune_val_results)))


            finetune_test_results = self.evaluate(model, test_data)
            print(f"[{self.name}][finetune][evaluation] test dataset: {finetune_test_results}")
            performance.append(("finetune", "test", str(finetune_test_results)))
        finally:
            df = pd.DataFrame(performance, columns=["stage", "dataset", "results"])
            df.to_csv(f"{self.callback_dir}/results.csv", index=False)

        print(f"[{self.name}][experiment] end")
        return model, pretrain_history, finetune_history
    
    def __str__(self):
        return self.name


"""
Task Types


"""


class RegressionTaskMixin:
    task_type = "regression"
    n_classes = 1

    def _get_top(self):
        return tf.keras.Sequential([
            get_default_top(),
            tf.keras.layers.Dense(1)
        ], name="top")

    def _compile_model(self, model, learning_rate, period):
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss="mae",
            metrics=[
                "mae",
                MeanAbsoluteErrorForYears(min_max_year=(1850, 1929)),
                AccuracyAtKYears(k=5, min_max_year=(1850, 1929)),
                AccuracyAtKYears(k=10, min_max_year=(1850, 1929)),
            ]
        )


class SparseClassificationMixin:
    task_type = "sparse"
    
    def _get_top(self):
        return tf.keras.Sequential([
            get_default_top(),
            tf.keras.layers.Dense(1)
        ], name="top")

    def _compile_model(self, model, learning_rate, period):
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "acc",
                SparseCategoricalAccuracyAtKYears(k=5, label_years=period),
                SparseCategoricalAccuracyAtKYears(k=10, label_years=period),
            ]
        )


class MulticlassClassificationTaskMixin:
    task_type = "multiclass_classification"

    def _get_top(self):
        return tf.keras.Sequential([
            get_default_top(),
            tf.keras.layers.Dense(self.n_classes)
        ], name="top")

    def _compile_model(self, model, learning_rate, period):
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=[
                "acc",
                CategoricalAccuracyAtKYears(k=5, label_years=period),
                CategoricalAccuracyAtKYears(k=10, label_years=period),
            ]
        )


class CORALTaskMixin:
    task_type = "CORAL"

    def _get_top(self):
        return tf.keras.Sequential([
            get_default_top(),
            coral.CoralOrdinal(self.n_classes)
        ], name="top")

    def _compile_model(self, model, learning_rate, period):
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss=coral.OrdinalCrossEntropy(self.n_classes),
            metrics=[
                coral.MeanAbsoluteErrorLabels(),
                "acc",
                # SparseCategoricalAccuracyAtKYears(k=5, label_years=period),
                # SparseCategoricalAccuracyAtKYears(k=10, label_years=period),
            ]
        )


"""
Feature Extraction

"""


class TFHubFeatureExtractionMixin:
    layer_url = None

    def _get_feature_extraction(self):
        return tf.keras.Sequential([
            hub.keras_layer.KerasLayer(self.layer_url)
        ], name="feature_extraction")


class ViTFeatureExtractionMixin(TFHubFeatureExtractionMixin):
    layer_url = "https://tfhub.dev/sayakpaul/vit_b32_fe/1"


class ResNet50V2FeatureExtractionMixin(TFHubFeatureExtractionMixin):
    layer_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5"


class SwinTransformerFeatureExtractionMixin(TFHubFeatureExtractionMixin):
    layer_url = "https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224_fe/1"


"""
Experiments

"""


class ResNetClassificationExperiment(ResNet50V2FeatureExtractionMixin,
                                     MulticlassClassificationTaskMixin, BaseExperiment):
    pass


class ResNetRegressionExperiment(ResNet50V2FeatureExtractionMixin,
                                 RegressionTaskMixin, BaseExperiment):
    pass


class CORALExperiment(ResNet50V2FeatureExtractionMixin, CORALTaskMixin, BaseExperiment):
    pass


class SwinTransformerExperiment(SwinTransformerFeatureExtractionMixin,
                                MulticlassClassificationTaskMixin, BaseExperiment):
    pass



if __name__ == "__main__":
    base_config = dict(name="test", n_classes=2)
    
    experiments = [
        ResNetClassificationExperiment(**base_config),
        ResNetRegressionExperiment(**base_config),
        CORALExperiment(**base_config),
        SwinTransformerExperiment(**base_config),
    ]
    
    for e in experiments:
        print(e)
        print(e.get_model().summary())
        
    experiments[0].run()
