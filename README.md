# Computer Science Thesis - Nicolas Hormann

This repository contains code and models from the NAACL 2022 paper [What kinds of errors do reference resolution models make and what can we learn from them?](https://aclanthology.org/2022.findings-naacl.152.pdf) by Jorge Sánchez, Mauricio Mazuecos, Hernán Maina and Luciana Benotti.

## 🔄 Updates and New Analysis
We've recently:
- Updated the codebase
- Integrated **[YAER](https://github.com/arielrossanigo/yaer)** as an experiment runner
- Conducted ablation studies analyzing how removing positional/visual information affects expression classification

## 🔍 Key Results
Our experiments reveal how information channels affect performance across expression types:

| Vis  | Pos  | Spatial       | Ordinal       | Relational    | Intrinsic     |
|------|------|---------------|---------------|---------------|---------------|
| ON   | ON   | 63.61 (0.38)  | 39.11 (3.08)  | 49.04 (0.58)  | 83.54 (0.35)  |
| ON   | OFF  | 55.10 (0.89)  | 30.00 (5.03)  | 38.42 (1.18)  | 77.91 (0.92)  |
| **Relative Diff. (%)** |  | **-13.37%** | **-23.30%** | **-21.66%** | **-6.73%**  |
| OFF  | ON   | 25.59 (0.55)  | 8.00 (2.65)   | 16.27 (1.06)  | 47.05 (0.36)  |
| **Relative Diff. (%)** |  | **-59.78%** | **-79.55%** | **-66.83%** | **-43.68%**  |
| OFF  | OFF  | 24.50 (0.68)  | 8.00 (1.45)   | 15.75 (0.78)  | 45.10 (0.50)  |
| **Relative Diff. (%)** |  | **-61.48%** | **-79.54%** | **-67.89%** | **-46.01%**  |

*Table 1: Accuracy (std. dev.) for ablation studies (threshold=0.5). Differences are relative to full-information baseline (Vis=ON, Pos=ON).*

## 📚 Further Exploration
- Detailed analysis: See Jupyter notebooks in [`notebooks/`](notebooks/)
- Experiment framework: [YAER repository](https://github.com/arielrossanigo/yaer)


## Installation & set up


1) Clone the repository.

```sh
$ git clone https://github.com/nhm-7/rec.git && cd rec
```

2) Environment. Any flavour of Conda. We recommend [miniconda](https://docs.conda.io/en/latest/miniconda.html). Use python 3.9 at least.

3) We created a environment.yml file. You need to run the following command:

```sh
$ conda env create -f environment.yml
$ conda activate rec-env
```

4) You'll also need a running version of [pytorch](https://pytorch.org/get-started/locally/). You can go to the website and choose the version that best suits your hardware and edit the requirements.txt according to that. Then, install all the requirements:

```sh
$ python3 -m pip install -r requirements.txt
```

## Setup data

Clone the [Referring Expression Dataset API](https://github.com/lichengunc/refer)

```sh
$ cd code/rec/
$ git clone https://github.com/lichengunc/refer.git && cd refer
$ git checkout python3
$ make
```

### Download data script

You can use the ```download_data.py``` script to download both refer and mscoco datasets. You just need to adapt the constants ```SETTINGS_REFER```, ```SETTINGS_MSCOCO``` and  ```SETTINGS_SAIAPR``` to your needs. It downloads those datasets and also unzip them to a custom path. We recommend to follow the data structure of the refer repository.


## Training and validation

First of all, all the experiments are configured with YAER, inside the code/rec/experiments/exps.py. So if you are thinking in writing a new experiment, you need to do there. Configure your experiment using the yaer decorator. Then, you need to be in the code/rec directory and run:

Run

```sh
$ yaer run -e <exp_name>
```

where <exp_name> is the name of the python function that you defined in the code/rec/experiments/exps.py file. The experiment will run and save the parameters, loggings and chekpoints by default into models/<exp_name>/ folder. We used custom scripts defined in tools/ directory to run them in mendieta. You can check and customize them to your needs. For more info about how to build and set up experiments using YAER, check the code/rec/experiments/README.md file.


## Pretrained models


[Here](https://drive.google.com/drive/folders/1ud7RaR_0rmJws4xGJeGz-tdZMugvd2eh?usp=sharing) you can find both the baseline and extended models trained on the different datasets (Table 3 in the paper). For convenience, we recommend to keep the same directory structure since the testing script infer some of the parameters from the path names.

* ReferItGame: [baseline](https://drive.google.com/drive/folders/1Yd0wVAGne5-drWz8wwlPjkIH6pZItzqm?usp=sharing), [extended](https://drive.google.com/drive/folders/1aPNzpfpeb0Y7Ztba-7N4EiR03LRqWzGg?usp=sharing)
* RefCOCO: [baseline](https://drive.google.com/drive/folders/1Zm92kg3ereWMSUqlqJocd9tG5dcI0U4y?usp=sharing), [extended](https://drive.google.com/drive/folders/1xTDmJzxJ_KbrmKj6DkBLqNyZtdkbcD6z?usp=sharing)
* RefCOCO+: [baseline](https://drive.google.com/drive/folders/1KxYomKbBTBEAWeB7DrnixwBavc44KZ3p?usp=sharing), [extended]()
* RefCOCOg: [baseline](https://drive.google.com/drive/folders/1YXw1Nt0gy34aaemOZJpigGvMq72Of2Zy?usp=sharing), [extended]()

Once you download the pretrained model, you need to locate it into the models/<pretrained_full_name_folder>, where <pretrained_full_name_folder> is the name that has the pretrained paper model as default. The predict submodule will infer the parameters from this <pretrained_full_name_folder>.

## Evaluation


First, you'll a running version of stanza. You can download the english package files as:

```sh
$ python3 -c "import stanza; stanza.download('en')"
```

You can also use spacy, in which case you need to change the ```backend="stanza"``` argument in line 178 to "backend=spacy". To get the spacy language files, run:

```sh
$ python3 -m spacy download en_core_web_md
```

Now, to test a trained model you need to be inside the code/rec/ folder and run any of the following commands(depending on if you use the paper models or those that you have trained using the YAER runner):

```sh
$ predict ~/models/<pretrained_full_name_folder>/best.ckpt --gpus <gpu_number>
$ predict ~/models/<exp_name>/best.ckpt --params ~/models/<exp_name>/params.log --gpus <gpu_number>
```

The script will infer the dataset and parameters from the <pretrained_full_name_folder> folder name, or will use the params.log ones. The test script is provided as an example use of our trained models. You can customize it to your needs.


## Contact


Any question do not hesitate to reach me (nicolas.hormann at mi.unc.edu.ar).