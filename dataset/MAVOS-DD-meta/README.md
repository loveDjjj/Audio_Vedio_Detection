---
language:
- ar
- ro
- en
- de
- hi
- es
- ru
task_categories:
- video-classification
---

LICENSE: This dataset is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) license.

This repository contains MAVOS-DD an open-set benchmark for multilingual audio-video deepfake detection.

Below, you can find the code to obtain the subsets described in the paper: train, validation, open-set model, open-set language and open-set full:

```{python}
from datasets import Dataset, concatenate_datasets
metadata = Dataset.load_from_disk('MAVOS-DD')
metadata_indomain = metadata.filter(lambda sample: sample['split']=='test' and not sample['open_set_model'] and not sample['open_set_language'])
metadata_open_model = metadata.filter(lambda sample: sample['split']=='test' and sample['open_set_model'] and not sample['open_set_language'])
metadata_open_model = concatenate_datasets([metadata_indomain, metadata_open_model])
metadata_open_language = metadata.filter(lambda sample: sample['split']=='test' and not sample['open_set_model'] and sample['open_set_language'])
metadata_open_model = concatenate_datasets([metadata_indomain, metadata_open_language])
metadata_all = metadata.filter(lambda sample: sample['split']=='test')
```

The scripts require the ``datasets`` package to be installed.
```{bash}
pip install datasets
```

We provide two scripts: metadata_generation.py and dataset.py.
The metadata_generation.py script is responsible for generating the metadata. Below is a sample metadata entry:
```{bash}
Sample: {'video_path': 'arabic/inswapper/02690.png_Po82BhllEjA_340_1.mp4.mp4', 'label': 'fake', 'split': 'train', 'open_set_model': False, 'open_set_language': False, 'language': 'arabic', 'generative_method': 'inswapper'}
```

The dataset.py script includes examples of how to read and filter this metadata.

The code for running the baseline models can be found here: 
https://github.com/CroitoruAlin/MAVOS-DD

Note:
Our dataset was collected from publicly available YouTube videos. If any individual wishes to request the removal of content involving them, please contact us at alincroitoru97@gmail.com.

Citation:
```{bash}
@misc{Croitoru-ArXiv-2025,
      title={MAVOS-DD: Multilingual Audio-Video Open-Set Deepfake Detection Benchmark}, 
      author={Florinel-Alin Croitoru and Vlad Hondru and Marius Popescu and Radu Tudor Ionescu and Fahad Shahbaz Khan and Mubarak Shah},
      year={2025},
      eprint={2505.11109},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.11109}, 
}
```