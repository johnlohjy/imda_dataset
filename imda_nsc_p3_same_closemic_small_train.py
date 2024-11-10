# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""

import csv
import json
import os

import datasets

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


# TODO: Add link to the official dataset URLs here
_DATA_URL = "data_train/imda_nsc_p3_small_train.tar.gz"

_PROMPTS_URLS = {
    "train": "data_train/prompts-train-small.txt.gz"
}

# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class SinglishDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
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
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        prompts_paths = dl_manager.download_and_extract(_PROMPTS_URLS) 
        archive = dl_manager.download(_DATA_URL) 
        train_dir = "../train"
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
    
    def _generate_examples(self, prompts_path, path_to_clips, audio_files):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        
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