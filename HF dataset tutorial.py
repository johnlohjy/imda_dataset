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

# Dataset loading script to manually create a dataset
# Audio datasets are commonly stored in tar.gz archives which requires a particular 
# approach to support streaming mode

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

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_DATA_URL = "data/vivos.tar.gz"

_PROMPTS_URLS = {
    "train": "data/prompts-train.txt.gz",
    "test": "data/prompts-test.txt.gz",
}

'''
On HF
- vivos.tar.gz
    - vivos.tar
        - train
            - genders.txt: Contains the gender type for each waves folder
            - prompts.txt: Contains transcriptions for all the .wav files
            - waves
                - VIVOSSPK01
                    - VIVOSSPK01_R001.wav
                    - VIVOSSPK01_R002.wav
                    - VIVOSSPK01_R003.wav
                    - ...
        - test
- prompts-train.txt.gz
    - prompts-train.txt: Contains transcriptions for all the .wav files
- prompts-test.txt.gz
'''

# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
# GeneratorBasedBuilder: Base class for datasets generated from a dictionary generator
class SinglishDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # Stores information about dataset like description, license and features
    def _info(self):
        return datasets.DatasetInfo(
            # This is the description of the dataset that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            # Specify dataset column types. Need to include Audio feature and sampling_rate 
            # of the dataset. Audio feature/column contains 3 important fields:
            #   array (decoded audio data as 1D array), path (to downloaded audio file), sampling_rate
            # https://huggingface.co/docs/datasets/en/about_dataset_features
            # Internal structure of the dataset. dict[column_name, column_type]
            features=datasets.Features(
                {
                    "speaker_id": datasets.Value("string"),
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "sentence": datasets.Value("string"),
                }
            ),
            supervised_keys=None, # What's this for???
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    # Downloads the dataset and defines its splits
    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        
        # Check on whether to use download_and_extract or download


        '''
        _DATA_URL = "data/vivos.tar.gz"
        _PROMPTS_URLS = {
            "train": "data/prompts-train.txt.gz",
            "test": "data/prompts-test.txt.gz",
        }
        On HF
        - vivos.tar.gz
            - vivos
                - train
                    - genders.txt: Contains the gender type for each waves folder
                    - prompts.txt: Contains transcriptions for all the .wav files
                    - waves
                        - VIVOSSPK01 -> Speaker ID
                            - VIVOSSPK01_R001.wav
                            - VIVOSSPK01_R002.wav
                            - VIVOSSPK01_R003.wav
                            - ...
                - test
        - prompts-train.txt.gz
            - prompts-train.txt: Contains transcriptions for all the .wav files
        - prompts-test.txt.gz
        '''
        # In streaming mode it doesn't download file(s), just returns URL to stream data from
        # Accepts a relative path to a file in the Hub dataset repo, example in data/
        prompts_paths = dl_manager.download_and_extract(_PROMPTS_URLS) # Download metadata file at _PROMPT_URLS
        archive = dl_manager.download(_DATA_URL) # Download audio TAR archive at _DATA_URL
        train_dir = "vivos/train"
        test_dir = "vivos/test"
        # Use SplitGenerator to organize audio files and transcriptions in each split 
        # All these file paths are passed to the next step to generate the dataset
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, # name each split
                gen_kwargs={ # These kwargs will be passed to _generate_examples
                    "prompts_path": prompts_paths["train"], # prompts-train.txt.gz
                    "path_to_clips": train_dir + "/waves", # vivos/train/waves
                    # Use iter_archive to iterate over audio files in the TAR archive
                    # This enables streaming for the dataset
                    "audio_files": dl_manager.iter_archive(archive), # archive contains vivos
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "prompts_path": prompts_paths["test"],
                    "path_to_clips": test_dir + "/waves",
                    "audio_files": dl_manager.iter_archive(archive),
                },
            ),
        ]
    # Generates the dataset's samples containing the audio data and other features specified in info
    # for each split
    # Yields a dataset according to the structure specified in features from the info method
    # It accepts: prompt_path, path_to_clips, audio_files (unpacked from gen_kwargs)
    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, prompts_path, path_to_clips, audio_files):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself. 
        
        # Files inside the TAR archives are accessed and yielded sequentially
        # Need to have the metadata associated with the audio files in the TAR file
        # so we can yield it with its corresponding audio file
        # Initialise our examples to be returned
        examples = {}
        # prompts_path: prompts-train.txt.gz
        with open(prompts_path, encoding="utf-8") as f:
            # For each transcription. Example: VIVOSSPK01_R001 transcription
            # Get the speaker id, audio file path, transcription and use the audio path as a key for the examples
            for row in f:
                data = row.strip().split(" ", 1) # Split into [VIVOSSPK01_R001, transcription]
                speaker_id = data[0].split("_")[0] # Get the speaker id VIVOSSPK01
                audio_path = "/".join([path_to_clips, speaker_id, data[0] + ".wav"]) # vivos/train/waves/VIVOSSPK01/VIVOSSPK01_R001.wav
                examples[audio_path] = {
                    "speaker_id": speaker_id,
                    "path": audio_path,
                    "sentence": data[1],
                }

        # Iterate over files in audio_files (vivos): Yield them along with their corresponding metadata
        # iter_archive() yields a tuple of (path, f) where path is a relative path to a file inside TAR archive 
        # and f is a file object itself.
        inside_clips_dir = False
        id_ = 0
        for path, f in audio_files:
            # path_to_clips example: vivos/train/waves/. If we are in the directory we want
            if path.startswith(path_to_clips):
                # Indicate that we are in the clips directory that we want
                inside_clips_dir = True
                # If the path is in the example dictionary we created earlier 
                if path in examples:
                    # add the audio information into the example: path and audio file in bytes
                    audio = {"path": path, "bytes": f.read()}
                    # Yield the id and example. Increment the id
                    yield id_, {**examples[path], "audio": audio}
                    id_ += 1
            # If we are here and inside_clip_dir is true it means we have finished processing the
            # directory that we want
            elif inside_clips_dir:
                break