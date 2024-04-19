## Installation

```
git clone https://github.com/yi1z/bitdelta.git
cd bitdelta
```

For Windows local machines:
```
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```
If you have an NVIDIA graphics card, make sure to install torch with cuda at [here](https://pytorch.org/get-started/locally/)

For Linux machines:
```
install anaconda from https://www.anaconda.com/

export PATH="/Users/()username)/anaconda3/bin:$PATH"

conda create --name csc413 python=3.9

conda activate csc413

(csc413) pip3 install -r requirements.txt
```

## Acknowledgments

```
@misc{liu2024bitdelta,
      title={BitDelta: Your Fine-Tune May Only Be Worth One Bit},
      author={James Liu and Guangxuan Xiao and Kai Li and Jason D. Lee and Song Han and Tri Dao and Tianle Cai},
      year={2024},
      eprint={2402.10193},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Model pairs used for compression testing
Base model: [DistilBERT base model](https://huggingface.co/distilbert/distilbert-base-uncased)

Finetuned models: [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

<br>

Base model: [DistilRoBERTa base](https://huggingface.co/distilbert/distilroberta-base)

Finetuned models: [DistilRoberta-financial-sentiment](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)

## Datasets used for compression testing
Sentiment Analysis: [GLUE](https://huggingface.co/datasets/nyu-mll/glue) (using sst2 subset), [GoEmotions](https://huggingface.co/datasets/go_emotions) (using simplified subset), [Stanford Sentiment Treebank](https://huggingface.co/datasets/stanfordnlp/sst2)

Other Tasks: [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank)

## Usage

```
python test.py
srun --gres gpu --partition=csc413 python3 test.py 
srun --gres gpu --partition=csc413 python3 test_by_name.py --base_model "google/fnet-large" --finetuned_model "gchhablani/fnet-large-finetuned-sst2" --dataset "glue" --subdata "sst2"
srun --gres gpu --partition=csc413 python3 test_by_name.py --base_model "distilbert/distilbert-base-uncased" --finetuned_model "anirudh21/distilbert-base-uncased-finetuned-rte" --dataset "glue" --subdata "rte"

```

## Demo

#Todo

## Triton

#Todo
