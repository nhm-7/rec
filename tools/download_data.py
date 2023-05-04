"""Download the data. Use python3 `download_data_from_refer.py -d all` to download all the data.
"""
import argparse
import requests
import pathlib
import os

from zipfile import ZipFile


EXT = '.zip'
SETTINGS_REFER = {
    "archive_base": "https://web.archive.org/web/",
    "bvision_base": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/",
    "datasets": ['refclef', 'refcoco', 'refcoco+', 'refcocog'],
    "keys": ['20220413011817', '20220413011718', '20220413011656', '20220413012904'],
    "unzip_path": pathlib.Path('./code/rec/refer/data/')
}
SETTINGS_MSCOCO = {
    "img_base": "http://images.cocodataset.org/zips/",
    "datasets": ['train2014', 'val2014', 'test2014'],
    "unzip_path": {'annotations': pathlib.Path('./code/rec/refer/data/mscoco/'),
                    'images': pathlib.Path('./code/rec/refer/data/images/mscoco/')}
}
SETTINGS_SAIAPR = {
    "archive_base": "https://web.archive.org/web/",
    "bvision_base": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/images/",
    "datasets": ['saiapr_tc-12'],
    "keys": ['20220515000000'],
    "unzip_path": pathlib.Path('./code/rec/refer/data/images/')
}


def _create_folder_if_needed(path: pathlib.Path) -> None:
    """Create a folder if it doesn't exist.

    :param path: The path of the folder to create
    :type path: pathlib.Path
    """
    path.mkdir(parents=True, exist_ok=True)


def _download(file_url: str, filename: str) -> None:
    """Download a file using filename as absolute path.

    :param file_url: The path from which download the file.
    :type file_url: str

    :param filename: filename of the downloaded file.
    :type filename: str
    """
    r = requests.get(file_url, stream = True)
    with open(filename, "wb") as my_file:
        for chunk in r.iter_content(chunk_size=1024):
            # writing one chunk at a time to my_file
            if chunk:
                my_file.write(chunk)


def _unzip(unzip_path: pathlib.Path, filename: str) -> None:
    """Unzip a file in a specific unzip_path.

    :param unzip_path: The path to unzip the data.
    :type unzip_path: pathlib.Path

    :param filename: filename of the downloaded file.
    :type filename: str
    """
    _create_folder_if_needed(unzip_path)
    with ZipFile(filename, 'r') as zipObj:
        zipObj.extractall(unzip_path)


def _download_ref(settings: dict) -> None:
    """Download referit datasets.

    :param settings: settings config to download the data.
    :type settings: dict
    """
    for dataset, key in zip(settings['datasets'], settings['keys']):
        print(f"downloading: {dataset}")
        filename = f'{dataset}{EXT}'
        file_url = os.path.join(
            settings['archive_base'], key, settings['bvision_base'], filename
        )
        _download(file_url, filename)
        print(f"{dataset} downloaded")
        _unzip(settings['unzip_path'], filename)
        print(f"{dataset} unzipped in {settings['unzip_path']} directory.")


def _download_mscoco(settings: dict) -> None:
    """Download mscoco datasets.

    :param settings: settings config to download the data.
    :type settings: dict
    """
    for dataset in settings['datasets']:
        print(f"downloading: {dataset}")
        filename = f'{dataset}{EXT}'
        file_url = os.path.join(settings['img_base'], filename)
        _download(file_url, filename)
        print(f"{dataset} downloaded")
        _unzip(settings['unzip_path']['images'], filename)
        print(f"{dataset} unzipped in {settings['unzip_path']['images']} directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=False, default='all')
    args, _ = parser.parse_known_args()
    data_to_download = args.data
    if data_to_download == 'all':
        _download_ref(SETTINGS_REFER)
        _download_mscoco(SETTINGS_MSCOCO)
        _download_ref(SETTINGS_SAIAPR)
    elif data_to_download == 'ref':
        _download_ref(SETTINGS_REFER)
    elif data_to_download == 'mscoco':
        _download_mscoco(SETTINGS_MSCOCO)
    elif data_to_download == 'saiapr':
        _download_ref(SETTINGS_SAIAPR)
    else:
        raise OSError("valid options are: 'all', 'ref', 'mscoco' and 'saiapr'")
