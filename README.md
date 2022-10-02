# What kinds of errors do reference resolution models make and what can we learn from them?

This repository contains code and models for our NAACL 2022 paper [What kinds of errors do reference resolution models make and what can we learn from them?]() by Jorge Sánchez, Mauricio Mazuecos, Hernán Maina and Luciana Benotti.

## Installation & set up


1) Clone the repository.

```sh
$ git clone https://github.com/nhm-7/rec.git && cd rec
```

2) Environment. Any flavour of Conda. We recommend [miniconda](https://docs.conda.io/en/latest/miniconda.html). Use python 3.9 at least.

3) We created a environment.yml file. You need to run the following command:

```sh
$ conda env create -f environments/environment.yml
$ conda activate rec-env
```

4) You'll also need a running version of [pytorch](https://pytorch.org/get-started/locally/). You can go to the website and choose the version that best suits your hardware. In our case, we installed as follows:

```sh
$ python3 -m pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch/
$ python3 -m pip install torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torchvision/
$ python3 -m pip install torchaudio==0.9.1 -f https://download.pytorch.org/whl/torchaudio/
```

5) Finally, install the requirements:

```sh
$ python3 -m pip install -r requirements.txt
```

## Setup data

Clone the [Referring Expression Dataset API](https://github.com/lichengunc/refer)

```sh
$ git clone https://github.com/lichengunc/refer.git && cd refer
$ git checkout python3
```

and follow the instructions to access the ReferItGame (a.k.a RefCLEF), RefCOCO, RefCOCO+ and RefCOCOg datasets.

### Download data script

You can use the ```download_data.py``` script to download both refer and mscoco datasets. You just need to adapt the constants ```SETTINGS_REFER``` and ```SETTINGS_MSCOCO``` to your needs. It downloads those datasets and also unzip them to a custom path. We recommend to follow the data structure of the refer repository.


## Training and validation

Run

```sh
$ python3 trainval.py -h
```

for a complete list of training options.


## Pretrained models


[Here](https://drive.google.com/drive/folders/1ud7RaR_0rmJws4xGJeGz-tdZMugvd2eh?usp=sharing) you can find both the baseline and extended models trained on the different datasets (Table 3 in the paper). For convenience, we recommend to keep the same directory structure since the testing script infer some of the parameters from the path names.

* ReferItGame: [baseline](https://drive.google.com/drive/folders/1Yd0wVAGne5-drWz8wwlPjkIH6pZItzqm?usp=sharing), [extended](https://drive.google.com/drive/folders/1aPNzpfpeb0Y7Ztba-7N4EiR03LRqWzGg?usp=sharing)
* RefCOCO: [baseline](https://drive.google.com/drive/folders/1Zm92kg3ereWMSUqlqJocd9tG5dcI0U4y?usp=sharing), [extended](https://drive.google.com/drive/folders/1xTDmJzxJ_KbrmKj6DkBLqNyZtdkbcD6z?usp=sharing)
* RefCOCO+: [baseline](https://drive.google.com/drive/folders/1KxYomKbBTBEAWeB7DrnixwBavc44KZ3p?usp=sharing), [extended]()
* RefCOCOg: [baseline](https://drive.google.com/drive/folders/1YXw1Nt0gy34aaemOZJpigGvMq72Of2Zy?usp=sharing), [extended]()


## Evaluation


First, you'll a running version of stanza. You can download the english package files as:

```sh
$ python3 -c "import stanza; stanza.download('en')"
```

You can also use spacy, in which case you need to change the ```backend="stanza"``` argument in line 178 to "backend=spacy". To get the spacy language files, run:

```sh
$ python3 -m spacy download en_core_web_md
```

Now, to test a trained model run:

```sh
$ python3 test.py <MODEL.ckpt>
```

The script will inferr the dataset and parameters from the file path. You can run ```-h``` and check which options are available. The test script is provided as an example use of our trained models. You can customize it to your needs.

# Error analysis annotation

We make available the annotation of the type of abilities needed for each RE to
be correctly resolved in the file
```ReferIt_Skill_annotation-NAACL2022.csv.g0```.

The file contains more type of abilities than the ones discussed in the paper.
The only types relevant for the analysis are:

 - fuzzy objects
 - meronimy
 - occlusion
 - directional
 - implicit
 - typo
 - viewpoint

## Citation

If you find this repository useful, please consider citing us.

```bibtex
@inproceedings{sanchez2022reference,
  title = {What kinds of errors do reference resolution models make and what can we learn from them?},
  author = {S\'anchez, Jorge and
    Mazuecos, Mauricio and
    Maina, Hern\'an and
    Benotti, Luciana},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {NAACL}},
  year = {2022},
  address = "Seattle, US",
  publisher = "{A}ssociation for {C}omputational {L}inguistics",
}
```


