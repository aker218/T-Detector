# T-Detector
Source Code of 《T-detector: A Trajectory based Pretained Model for Game Bot Detection in MMORPGs》

## Install dependencies
Base environment：python3.6

Packages：
    
    matplotlib==3.1.1
    numpy==1.18.1
    pandas==0.25.2
    Cython==0.29.23
    gensim==3.8.2
    torch==1.1.0
    scipy==1.5.1
    tqdm==4.37.0
    dgl==0.4.1
    scikit_learn==0.24.2
    transformers==2.5.1

Under the corresponding Python version, run requirements.txt in /trajectory_detector/

    pip install -r requirements.txt
    
## Directory description

```shell
trajectory_detector
├── data                                     # data dir
├── models                                   # model dir
├── requirements.txt                         # install package
├── dataset.py                               # torch dataset function
├── models.py                                # torch model class
├── trainer.py                               # train and evaluate function
├── preprocess.py                            # raw data preprocess
├── time_dis_w2v_preprocess.py               # preprocess of LocationTime2Vec
├── time_dis_w2v.py                          # LocationTime2Vec
├── w2v.py                                   # Word2Vec
├── make_dataset.py                          # Convert processed data to dataset
├── angle_pretrain.py                        # Angle Pretrain
├── train_and_evaluate.py                    # Model Training and Evaluation
├── run.sh                                   # Automated execution script
```

## Dataset preparation
The dataset dir should be named "new_dataset/" and placed in the /trajectory_detector/data/，the structure of new_dataset/ is as follows:
```shell
├── label.csv                                # label file
├── move                                     # game character trajectory data dir
│   ├── 2f8ea2aeaf01249c02c66cb652a723a3_16_2021-05-10.json 
│   └── 357fe4456ef4a536b2112daddb347a0b_14_2021-05-10.json 
│   └── ......
├── mouse                                     # mouse trajectory data dir
│   ├── 2f8ea2aeaf01249c02c66cb652a723a3_16_2021-05-10.json 
│   └── 357fe4456ef4a536b2112daddb347a0b_14_2021-05-10.json
│   └── ......
```
label.csv is shown in follows, the id consists of the "user id_map id_sample date" and matches the file names in move/ and mouse/:
| id	     | label | 
| :--------|:------|
|c8b7023594c144a3421dafa07c9d4c53_0_2021-05-15| 0|
|673aa532d8ed3e1e7926d16ac37328ee_3_2021-05-15| 1|
|... | ... |

the sample in move/ and mouse/ is shown in follows:

```
[
    {
        "x": 24182,
        "y": 5938,
        "tm": 1599049877072
    },
    {
        "x": 24182,
        "y": 5938,
        "tm": 1599049877075
    },
    {
        "x": 24182,
        "y": 5938,
        "tm": 1599049877117
    }
    ……
]
```

- Field Description: tm: millisecond timestamp, x: map x, y: map y

## Data preprocess->Embedding pretrain->Angle pretrain->Model training and evaluation
After data preparation, directly execute the run.sh and complete all the steps, among which the embedding pre-training step and Angle pretrain are slow and need to wait for 2-4 hours. Upon completion of the execution, all model parameters and test set evaluation results will be stored under /trajectory_detector/models/.
    
    source run.sh
