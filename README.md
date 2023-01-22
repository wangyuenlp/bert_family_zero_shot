# Are the BERT Family Zero-Shot Learners? A Study on Their Potential and Limitations.
This repo contains the code of the paper:  Are the BERT Family Zero-Shot Learners? A Study on Their Potential and Limitations.




## How to use
### 1. Requirements
Please run the following script to install the remaining dependencies first.

```
pip install -r requirements.txt
```

### 2. Dataset and Models

We implement the code based on https://github.com/huggingface/transformers and https://github.com/huggingface/datasets.

Therefore, datasets and models will be automatically downloaded if you can connect to the Internet.

### 3. Experiments

To reporduce the results of Multi Null Prompt, please run the following script:

```
cd multinull
bash train.sh
```
To reporduce the results of Dynamic Prompt, please run the following script:

```
cd dynamicmask
bash train.sh
```

To reporduce the results of Self Generate Prompt, please run the following script:

```
cd selfgenerate
bash train.sh
```

