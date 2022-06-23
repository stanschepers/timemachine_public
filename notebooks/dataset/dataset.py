import os
import time

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

def to_5_years(year: int) -> str:
    if year is None:
        return "None"
    century, decade, unit = str(year)[:2], str(year)[2], str(year)[3]
    if int(unit) < 5:
        return f"{century}{decade}0-4"
    return f"{century}{decade}5-9"


def load_dataset(dataset_dir, y_col="year", class_mode="raw", validation_split=0.2, batch_size=32, color_mode="rgb",
                 target_size=(224, 224), train_preprocess_config=None, test_preprocess_config=None, df: pd.DataFrame = None):
    print(f"[dataset][{dataset_dir}] start loading")
    start_time = time.time()

    if train_preprocess_config is None:
        train_preprocess_config = dict()
    if test_preprocess_config is None:
        test_preprocess_config = dict()
    
    if df is None:
        df = pd.read_csv(os.path.join(dataset_dir, "info.csv"))

    generator_config = {
        "directory": dataset_dir,
        "x_col": "filename",
        "y_col": y_col,
        "class_mode": class_mode,
        "color_mode": color_mode,
        "batch_size": batch_size,
        "target_size": target_size
    }

    train_preprocess_config["validation_split"] = validation_split

    train_datagen = ImageDataGenerator(**train_preprocess_config)
    test_datagen = ImageDataGenerator(**test_preprocess_config)

    train_generator = train_datagen.flow_from_dataframe(df[df["set"] == "train"], subset="training", **generator_config)
    val_generator = train_datagen.flow_from_dataframe(df[df["set"] == "train"], subset="validation", **generator_config)
    try:
        test_generator = test_datagen.flow_from_dataframe(df[df["set"] == "test"], **generator_config)
    except KeyError:
        print(f"[dataset][{dataset_dir}] skipping test dataset")
        test_generator = None
              

    print(f"[dataset][{dataset_dir}] end loading in", round(time.time() - start_time, 2), "s")
    return train_generator, val_generator, test_generator


def load_dataset_info(dataset_dir):
    return pd.read_csv(os.path.join(dataset_dir, "info.csv"))

