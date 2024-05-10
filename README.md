# Zero-shot urban function inference with street view images through prompting a pre-trained vision-language model

This project is linked to a paper accepted to International Journal of Geographical Information Science.
In this study, we develop the prompting framework UrbanCLIP, which builts upon the vision-language pretrained model CLIP, to carry out zero-shot urban function inference using street view images (SVIs).

## Quick start
With this repository, you can 
- reproduce the results in the paper
- use the shared urban scene dataset for further studies
- potentially carry out zero-shot urban function inference using your own SVIs


## Structure
The structure of this repository is as follows:
- `Zeroshot_UrbanCLIP.py`: the main portal for zero-shot urban function inference with SVIs
- `./Data`: the folder to store the data, including the annotated urban scenes (SVIs) in Shenzhen, Singapore, and London.
- `./Emb`: the folder to store the SVI embeddings.
- `./Utils`: the folder to store the urban taxonomy and urban fucntion prompts, as well as some utility functions.


## `Zeroshot_UrbanCLIP.py`
`Zeroshot_UrbanCLIP.py` is used to carry out zero-shot urban function inference with SVIs. You could specify several arguments in `zeroshot_inference.py`:

- `--task`: can be "primary", "multi", "transfer-singapore" or "transfer-london", to reproduce the results in the paper.
- `--taxomony`: indicate if the developed urban taxomony is to be used, with the option "UrbanCLIP" indicting the use of the urban taxonomy, and "function_name" indicting otherwise.
- `--prompt_template`: the prompt templates to use, which can be UrbanCLIP, Wu, Photo, CLIP80, no_template, UrbanCLIP_SC, Wu_without_SC, and please refer to the paper for more details.
- `--ensemble`: indicate the prompting template ensembling method, which can be "mean" or "zpe".
- `--device`: the device to use, which can be `cpu` or `cuda` or `cuda:{}`.

For example, you could run the command: 

`python Zeroshot_UrbanCLIP.py --task=primary --device=cuda:0` 

to reproduce the results in the paper on zero-shot primary function inference.

## `contact`

Weiming Huang

Email: weiming.huang@nateko.lu.se 
