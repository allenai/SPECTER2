# SPECTER2

**Dec 2023 Update:**

Model usage updated to be compatible with latest versions of transformers and adapters (newly released update to adapter-transformers) libraries.

**Aug 2023 Update:**
1. The SPECTER2 Base and proximity adapter models have been renamed in Hugging Face based upon usage patterns as follows:

|Old Name|New Name|
|--|--|
|allenai/specter2|[allenai/specter2_base](https://huggingface.co/allenai/specter2_base)|
|allenai/specter2_proximity|[allenai/specter2](https://huggingface.co/allenai/specter2)|

2. We have a parallel version (termed [aug2023refresh](https://huggingface.co/allenai/specter2_aug2023refresh)) where the base transformer encoder version is pre-trained on a collection of newer papers (published after 2018).
   However, for benchmarking purposes, please continue using the current version.

## Overview
SPECTER2 is a collection of document embedding models for scientific tasks. It builds on the original [SPECTER](https://github.com/allenai/specter) and [SciRepEval](https://github.com/allenai/scirepeval) works, and can be used to generate specific embeddings for multiple task formats i.e Classification, Regression, Retrieval and Search based on the chosen type of associated adapter (examples below). 

**Note:** To get the best performance for a particular task format, please load the appropriate adapter along with the base transformer model as given [below](https://github.com/allenai/SPECTER2_0#huggingface). 

## Setup
If using the existing model weights for inference:
```bash
https://github.com/allenai/SPECTER2.git
cd SPECTER2
conda create -n specter2 python=3.8
pip install -e .
pip install -r requirements.txt

# or uv sync --extra cpu
```

For training/ benchmarking, please setup [SciRepEval](https://github.com/allenai/scirepeval)

## Usage
We train a base model from scratch on citation links like SPECTER, but our training data consists of 6M (10x) triplets spanning 23 [fields of studies](https://api.semanticscholar.org/CorpusID:256194545). 
Then we train task format specific adapters with SciRepEval to generate multiple embeddings for the same paper.
We represent the input paper as a concatenation of its title and abstract.
For Search type tasks where the input query is a short text rather a paper, use the adhoc query model below to encode it and the retrieval model to encode the candidate papers.  
All the models are publicly available on HuggingFace and AWS S3.

### HuggingFace
|Model|Name and HF link|Description|
|--|--|--|
|Retrieval*|[allenai/specter2_proximity](https://huggingface.co/allenai/specter2)|Encode papers as queries and candidates eg. Link Prediction, Nearest Neighbor Search|
|Adhoc Query|[allenai/specter2_adhoc_query](https://huggingface.co/allenai/specter2_adhoc_query)|Encode short raw text queries for search tasks. (Candidate papers can be encoded with proximity)|
|Classification|[allenai/specter2_classification](https://huggingface.co/allenai/specter2_classification)|Encode papers to feed into linear classifiers as features|
|Regression|[allenai/specter2_regression](https://huggingface.co/allenai/specter2_regression)|Encode papers to feed into linear regressors as features|

*Retrieval model should suffice for any other downstream task types not mentioned above

```python
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')

#load base model
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
#other possibilities: allenai/specter2_<classification|regression|adhoc_query>

papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
          {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]

# concatenate title and abstract
text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
# preprocess the input
inputs = self.tokenizer(text_batch, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=512)
output = model(**inputs)
# take the first token in the batch as the embedding
embeddings = output.last_hidden_state[:, 0, :]
```

### AWS S3 via CLI
```bash
mkdir -p specter2/models
cd specter2/models
aws s3 --no-sign-request cp s3://ai2-s2-research-public/specter2_0/specter2_0.tar.gz .
tar -xvf specter2_0.tar.gz
```
The above commands will copy all the model weights from S3 as a tar archive and extract two folders-base and adapters.

```python
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('specter2/models/base')

#load base model
model = AutoAdapterModel.from_pretrained('specter2/models/base')

#load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
model.load_adapter("specter2/models/adapters/proximity", load_as="proximity", set_active=True) 
#other possibilities: .../adapters/<classification|regression|adhoc_query>

papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
          {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]

# concatenate title and abstract
text_batch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
# preprocess the input
inputs = self.tokenizer(text_batch, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=512)
output = model(**inputs)
# take the first token in the batch as the embedding
embeddings = output.last_hidden_state[:, 0, :]
```

### Batch Processing for multiple task types (requires GPU)
To generate the embeddings for an input batch, follow [INFERENCE.md](https://github.com/allenai/scirepeval/blob/main/evaluation/INFERENCE.md).
Create the Model instance as follows:
```python

adapters_dict = {"[CLF]": "allenai/specter2_classification", "[QRY]": "allenai/specter2_adhoc_query", "[RGN]": "allenai/specter2_regression", "[PRX]": "allenai/specter2_proximity"}
model = Model(variant="adapters", base_checkpoint="allenai/specter2", adapters_load_from=adapters_dict, all_tasks=["[CLF]", "[QRY]", "[RGN]", "[PRX]"])
```
Follow Step 2 onwards in the provided ReadMe.

## Training

The training and validation triplets have been added to the SciRepEval benchmark, and is available [here](https://huggingface.co/datasets/allenai/scirepeval/viewer/cite_prediction_new/evaluation).
The training data consists of triplets from [SciNCL](https://github.com/malteos/scincl) as a subset.

The training triplets cover the following fields of study:

|Field of Study|
|-|
|Agricultural And Food Sciences|
|Art|
|Biology|
|Business|
|Chemistry|
|Computer Science|
|Economics|
|Education|
|Engineering|
|Environmental Science|
|Geography|
|Geology|
|History|
|Law|
|Linguistics|
|Materials Science|
|Mathematics|
|Medicine|
|Philosophy|
|Physics|
|Political Science|
|Psychology|
|Sociology|

The model is trained in two stages using [SciRepEval](https://github.com/allenai/scirepeval/blob/main/training/TRAINING.md):
- Base Model: First a base model is trained on the above citation triplets.
``` batch size = 1024, max input length = 512, learning rate = 2e-5, epochs = 2```
- Adapters: Thereafter, task format specific adapters are trained on the SciRepEval training tasks, where 600K triplets are sampled from above and added to the training data as well.
``` batch size = 256, max input length = 512, learning rate = 1e-4, epochs = 6```


## Evaluation
We evaluate the model on [SciRepEval](https://github.com/allenai/scirepeval), a large scale eval benchmark for scientific embedding tasks which which has [SciDocs] as a subset.
We also evaluate and establish a new SoTA on [MDCR](https://github.com/zoranmedic/mdcr), a large scale citation recommendation benchmark.

|Model|SciRepEval In-Train|SciRepEval Out-of-Train|SciRepEval Avg|MDCR(MAP, Recall@5)|
|--|--|--|--|--|
|[BM-25](https://api.semanticscholar.org/CorpusID:252199740)|n/a|n/a|n/a|(33.7, 28.5)|
|[SPECTER](https://huggingface.co/allenai/specter)|54.7|72.0|67.5|(30.6, 25.5)|
|[SciNCL](https://huggingface.co/malteos/scincl)|55.6|73.4|68.8|(32.6, 27.3)|
|[SciRepEval-Adapters](https://huggingface.co/models?search=scirepeval)|61.9|73.8|70.7|(35.3, 29.6)|
|[SPECTER2 Base](allenai/specter2_base)|56.3|73.6|69.1|(38.0, 32.4)|
|[SPECTER2-Adapters](https://huggingface.co/models?search=allenai/specter-2)|**62.3**|**74.1**|**71.1**|**(38.4, 33.0)**|

The per task evaluation result can be found in this [spreadsheet](https://docs.google.com/spreadsheets/d/1HKOeWYh6KTZ_b8OM9gHfOtI8cCV9rO1DQUjiUIYYcwg/edit?usp=sharing).

## Citation
Please cite the following works if you end up using SPECTER 2.0:
[SciRepEval paper](https://api.semanticscholar.org/CorpusID:254018137)
```bibtex
@inproceedings{Singh2022SciRepEvalAM,
  title={SciRepEval: A Multi-Format Benchmark for Scientific Document Representations},
  author={Amanpreet Singh and Mike D'Arcy and Arman Cohan and Doug Downey and Sergey Feldman},
  booktitle={Conference on Empirical Methods in Natural Language Processing},
  year={2022},
  url={https://api.semanticscholar.org/CorpusID:254018137}
}
```
