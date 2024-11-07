import csv
import json
import os

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

_HOMEPAGE = ""

_LICENSE = ""

_DATA_URL = "data/imda_nsc_p3.tar.gz"

_PROMPTS_URLS = {
    "train": "data/prompts-train.txt.gz",
}

class SinglishDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "sentence": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        prompts_paths = dl_manager.download_and_extract(_PROMPTS_URLS) 
        archive = dl_manager.download(_DATA_URL) 
        train_dir = "train"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={ 
                    "prompts_path": prompts_paths["train"], 
                    "path_to_clips": train_dir + "/waves", 
                    "audio_files": dl_manager.iter_archive(archive), 
                },
            )
        ]
    
    '''
    compressing gzip on the top level is enough
    '''


    def _generate_examples(self, prompts_path, path_to_clips, audio_files):
        examples = {}
        with open(prompts_path, encoding="utf-8") as f:
            for row in f:
                data = row.strip().split(" ", 1) 
                audio_path = "/".join([path_to_clips, data[0] + ".wav"]) 
                examples[audio_path] = {
                    "path": audio_path,
                    "sentence": data[1],
                }

        inside_clips_dir = False
        id_ = 0
        for path, f in audio_files:
            if path.startswith(path_to_clips):
                inside_clips_dir = True
                if path in examples:
                    audio = {"path": path, "bytes": f.read()}
                    yield id_, {**examples[path], "audio": audio}
                    id_ += 1
            elif inside_clips_dir:
                break