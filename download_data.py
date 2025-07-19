import json
import os

CREDS_PATH = './.kaggle/kaggle.json'

with open(CREDS_PATH) as fp:
    creds = json.load(fp)

for k, v in creds.items():
    os.environ[f'KAGGLE_{k}'] = v

# Authenticate using kaggle.json
import kaggle  # noqa
kaggle.api.authenticate()

# Download a dataset
kaggle.api.dataset_download_files(
    'anggadwisunarto/text-deblurring-dataset-with-psf-for-ocr',
    path='./data', unzip=True, quiet=False
)
