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

## Datasets used for compression testing
Sentiment Analysis: [GLUE](https://huggingface.co/datasets/nyu-mll/glue)

## Usage

```
python test.py
srun --gres gpu --partition=csc413 python3 test.py 

```

## Demo
```
If you want to pass in multiple fine-tuned models that share the same base model, but might train on different dataset, please refer to bitdelta/ft_model.json and follow the format.
```

## [Final Paper](https://github.com/alyssachne/bitdelta_pipeline/blob/deploy/CSC413.pdf)

