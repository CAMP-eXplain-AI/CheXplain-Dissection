import os
from src.__init__ import ROOT_DIR
from torchvision import transforms as torch_transforms
from PIL import Image
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

def preprocess_image(path, path_to_data, transform):
    """ Preprocesses a dataset and saves it to disk

    Args:
        path (str): Path to the image
        path_to_data (str): Path to the dataset
        transform (torchvision.Transforms): Torchvision transform object
    """

    path = os.path.join(path_to_data, path)
    img = Image.open(path).convert('RGB')

    img = transform(img)

    dest_path = path
    idx = dest_path.find('view')
    dest_path = dest_path[: idx-1]
    dest_path = dest_path.replace('dataset/', 'dataset_processed/')

    os.makedirs(dest_path, exist_ok=True)

    dest_path = dest_path + path[idx-1:]
    img.save(dest_path)


if __name__ == '__main__':

    print('Started preprossesing of CheXpert')

    path = os.path.join(ROOT_DIR, 'data/chexpert/dataset')
    size = (320,320)
    processes = 14

    transform = torch_transforms.Compose([

    ])

    fold = ['train', 'valid', 'test']

    # Iterate over folds and save image
    for csv in fold:

        print(f'Current fold: {fold}.')

        df = pd.read_csv(os.path.join(path, (csv + ".csv")))

        path_df = np.asarray(df['Path'])

        pool = Pool(processes=processes)

        wrapper = partial(preprocess_image, path_to_data=path, transform=transform)

        result = pool.map_async(wrapper, path_df)
        result.get()
        
    print('Finished preprocessing.')
