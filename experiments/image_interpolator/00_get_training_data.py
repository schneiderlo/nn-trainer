import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sys import getsizeof

from nn_trainer.data.item_handler import NPArrayHandler
from nn_trainer.data.data_model import DataModel
from nn_trainer.data.data_augmentation import ConversionOp
from nn_trainer.utils.image import imread
from nn_trainer.utils.tests import get_test_directory, get_test_workspace


DATA_MODEL = DataModel(
  item_handlers=[
    NPArrayHandler(
      'src',
      np_dtype=np.uint8,
      data_augmentation_ops=[ConversionOp(output_tf_type=tf.float32)]
    ),
  ]
)


def get_initial_df() -> pd.DataFrame:
  input_csv = ""
  df = pd.read_csv(input_csv, sep=";")
  df = df[~df['DATASET_LABEL'].str.contains("RIG")]
  df = df.rename(columns={
    'SEM_IMAGE': 'src',
  })
  df = df[['src']]
  return df


def test_convert_csv(request):
  output_dir = get_test_directory(request)
  output_csv = os.path.join(output_dir, "sem_db.csv")
  df = get_initial_df()
  df.to_csv(output_csv, index=False)
  total_size = sum([getsizeof(imread(src_path, dtype=np.uint8))
                    for src_path in df['src'].to_list()])
  print("\nNumber of images: {}".format(len(df)))
  print("Size on disk: {:.2f} GB".format(total_size * 1e-9))


def test_create_tfrecords(request):
  output_dir = get_test_directory(request)
  input_csv = get_test_workspace("test_convert_csv", "sem_db.csv")
  df = pd.read_csv(input_csv)
  DATA_MODEL.export_to_tfrecord(
    output_dir=output_dir,
    dataframe=df,
    max_size=2000
  )
