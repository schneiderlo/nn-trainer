import os
import pandas as pd

from nn_trainer.utils.tests import get_test_directory


def test_convert_csv(request):
  output_dir = get_test_directory(request)
  input_csv = "/data/storage/projets/Nuflare-Inspection/ASELTA/work/loic/database.csv"
  df = pd.read_csv(input_csv, sep=";")
  df = df.rename(columns={
    'SEM_IMAGE': 'src',
    'lscad_raster': 'dist_to_ctr'
  })
  df = df[['src', 'dist_to_ctr']]
  output_csv = os.path.join(output_dir, "sem_db.csv")
  df.to_csv(output_csv, index=False)
