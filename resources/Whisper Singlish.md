# Tuning Whisper for Singlish

Resources
- https://www.jensenlwt.com/blog/singlish-whisper-finetuning-asr-for-singapore-unique-english
- https://medium.com/htx-dsai/finetuning-whisper-for-the-singaporean-home-team-context-a3ae1a6ae809

**<u>Original Whisper AI:</u>**

<u>Multilingual Support</u>

The models (there are 5) have been trained on 680,000 hours of audio in 99 languages (of which 16 are South Asian, 10 Southeast Asian) with corresponding transcripts downloaded from the internet. About 65% of this data is English audio, 18% is non-English audio with English transcripts, and 17% is non-English audio.

Whisper is trained on a large, diverse, multi-lingual dataset consisting of 680,000 hours of audio collected from the web. Thanks to the diversity in its dataset, with more than one-third of the data coming from non-English languages, Whisper is able to generalize better to different languages.

Languages supported by Whisper, with a focus on languages of [South Asia (16) and South East Asia (10)](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py): Assamese, Bengali, Chinese, English, Gujarati, Hindi, Indonesian, Javanese, Kannada, Khmer, Lao, Malay, Malayalam, Marathi, Myanmar, Nepali, Pashto, Punjabi, Sanskrit, Sindhi, Sinhala, Sundanese, Tagalog, Tamil, Telugu, Thai, Urdu, Vietnamese

Still struggles with a low-resource languages. Does it struggle with Singlish?

<br/>
<br/>
<br/>

<u>**Data**</u>

IMDA National Speech Corpus

Consists of 6 parts of recordings of local english speakers
- Prompted recordings of phonetically-balanced scripts
- Prompted recordings of sentences randomly generated from words based on people, food, location, brands
- Part 3: Conversational data on topics covering daily life and playing games provided. 1000 hours of conversational speech
- Conversational code-switched data (from Singaporean English to various native languages)
- Conversational data on debate, finance topics, with positive and negative emotion
- Conversational data in 3 styles (holiday/restaurant/hotel, bank/telephone/insurance, HDB/MOE/MSF).

<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

<u>**Home Team's Approach**</u>

**Data Stuff**

Home Team used NSC Part 3 as it was determined to best suit their use case due to its conversational nature,
whereas Part 1 and 2 were focused on phrase level utterances

They streamed training data - load data progressively while iterating over the dataset. Stream only the audio samples that they need throughout the training process
- https://huggingface.co/blog/audio-datasets
- https://huggingface.co/docs/datasets/en/stream

They adapted the dataset template and wrote their own custom loading script for NSC Part 3
- https://github.com/huggingface/datasets/blob/main/templates/new_dataset_script.py 
- https://huggingface.co/docs/datasets/en/audio_dataset#loading-script

Loading a dataset in streaming mode creates IterableDataset objects that they can then use for finetuning

```
def prepare_dataset(batch):
    # load the audio 
    audio = batch["audio"]
    # perform feature extraction by computing log-Mel input features from input audio
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # tokenize the transcripts (convert text into numerical representations that the model can use) 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

# use the custom loading script to load the data as an IterableDataset with streaming
imda3_train = load_dataset("local_loadingScript_imda_part3.py","all", split='train', streaming=True)
imda3_val = load_dataset("local_loadingScript_imda_part3.py","all", split='validation', streaming=True)
    
imda_dataset = IterableDatasetDict()
imda_dataset["train"] = imda3_train
imda_dataset["val"] = imda3_val

imda_processed = imda_dataset.map(prepare_dataset, remove_columns=next(iter(imda_dataset.values())).column_names)
```

The NSC Part 3 recordings are split into two environments, each with two different microphones used for recording. In the first environment, where speakers were in the same room, we selected the recordings using the close-talk mic as this isolated the main speaker’s voice (without picking up background noise or the secondary speaker). For the second environment with speakers in different rooms, we chose to use the standing microphone recordings, as opposed to recordings via telephone.
- consists of WAV files and corresponding TextGrid transcript files 
- each audio clip is 1hr long 

The preprocessing steps they did were as follows:

1. Clean transcripts via removing annotations for
- Paralinguistic phenomena (e.g., breathing, coughing, laughing) — this is represented in the text by annotations such as (ppo), (ppb), (ppl) etc.
- Fillers or unknown words ```<FIL/>```, unclear words ```<UNK>```, short pauses ```<S>``` etc. according to the NSC transcription guidelines
- Unique Singlish particles, we removed the annotations and kept the particles as part of the text e.g., ‘ok ```[lah]``` we go there’ → ‘ok lah we go there’

2. Normalise transcript text
- To avoid inconsistent casings and punctuation in the output of the final fine-tuned model, remove punctuations and lower-case the text

3. Create 30s audio segments with their corresponding transcripts for training
- Using the time segments from the TextGrid files we splice out the corresponding segments from the WAV files. Shorter consecutive segments were also combined, up to a length of 30s. This corresponds to Whisper’s intrinsic design where its feature extractor ensures all segments have a length of 30s before training

20 hours of data is used for validation

Another 20 hours of data is used as a test set 

WER is used for evaluation 

Create a custom data collator that pads audio features and tokenized labels to the appropriate max length
- Converted 30s audio chunks and corresponding transcripts into audio features and tokenized labels 

```
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # first treat the audio inputs by returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # remove begining-of-sequence (bos) tokens to remove redundancy as they are added later when needed
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
```

This code processes inputs and labels separately

Padding tokens are replaced wih -100 so they are ignored in loss calculations during finetuning

**Fine-tuning stuff**

In order to monitor the model’s performance more effectively, we can also define a custom WER metric to use during evaluation

```
import evaluate 
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the original pad_token_id (since we had replaced padding in the DataCollator above) 
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # convert the predictions and labels back into human readable text, ignoring special tokens 
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```

Specify training arguments:
- Note: Learning rate is 40x smaller than what authors of Whisper paper suggested
- Should experiment with specific parameters for own fine-tuning use case 

```
# Specify the base version of Whisper to fine-tune
model_name_base = "openai/whisper-medium"
model = WhisperForConditionalGeneration.from_pretrained(model_name_base)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

# Use the custom data collator we defined above!
processor = WhisperProcessor.from_pretrained(model_name_base, language="English", task="transcribe")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor) 

# Define the training arguments
job_name = 'htx-medium-nscpart3'
training_args = Seq2SeqTrainingArguments(
    output_dir=f"/datadrive/{job_name}",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=6.25e-6, # 40x smaller that what was used for pretraining
    warmup_steps=300,
    max_steps=3000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=50,
    eval_steps=50, #smaller than usual, for initial experimentation
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer", # from our custom compute_metrics function above!
    greater_is_better=False,
)

# Create the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=imda_processed["train"],
    eval_dataset=imda_processed["val"],
    data_collator=data_collator, 
    compute_metrics=compute_metrics, # we defined the custom compute_metrics function above!
    tokenizer=processor.feature_extractor,
)
```

Saved a model checkpoint and evaluated the model every 50 steps, which provided us with a fine-grained view of our training. However, this results in a longer overall finetuning duration, since the model is being evaluated constantly. We can consider increasing the save and evaluation steps for future training iterations.

The team initially configured training settings to finetune the model on 3000 steps, while checkpointing and evaluating the model every 50 steps. However, based on the evaluation and loss curves they early stop the training after 1000 steps.
...

**Results**

- whisper-med (Baseline): 28.2% WER
- whisper-med (finetuned NSC part 3): 14.5% WER

**Other Notes**

Rare words may not be represented well within the model's learnt vocabulary 
- In order to decode efficiently, end-to-end models often narrow down the most likely output sequence through beam search, and rare context-dependent words or phrases may not get captured.
- To overcome this, for everyday local words and phrases, a dataset like NSC Part 2 (containing sentences randomly generated from words based on people, food, location and brands) may be more appropriate for finetuning
- Combine datasets 
- Or a subsequent fine-tuning stage 

Catastrophic forgetting may also happen because of finetuning
- degrades Whisper’s existing multilingual or translation capabilities



<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

<u>**Approach by [Jensen lwt](https://www.jensenlwt.com/blog/singlish-whisper-finetuning-asr-for-singapore-unique-english)**</u>

whisper-small model

use 2 subsets of the data (may not fully capture the entrie diversity of local language) to fine-tune whisper to see how the amount of data affects the model's performance

Subset 1
- 39 661 samples, 52 hours. 80/10/10 (train/valid/test) split

Subset 2
- 122 142 samples, 163 hours. 80/10/10 split. Approximately 16% of available data

Baseline performance by whisper-small on held-out test set using benchmark metric WER: 57.9%

**Initialise Whisper**
```
# https://huggingface.co/docs/transformers/v4.38.2/en/model_doc/whisper#transformers.WhisperForConditionalGeneration
# https://github.com/huggingface/transformers/issues/29394

# Bare Whisper model has no specific head on top - outputs raw hidden-states 
# WhisperForConditionalGeneration has a LM head, can be used for ASR
from transformers import WhisperForConditionalGeneration


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# disable cache during training since it's incompatible with gradient checkpointing
# https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958/2
# Only used inference because we have teacher forcing 
model.config.use_cache = False

# set language and task for generation and re-enable cache
# modify the generate method of the model by setting some default values
# use_cache needs to be T
"""
In transformer architectures, each layer's self-attention mechanism computes key (`K`) and value (`V`) pairs based on the input tokens. These pairs are vital for determining how each token in a sequence relates to every other token, influencing the final output generated after processing the sequence.

When processing sequences incrementally, as in this case where tokens are fed one-by-one, retaining the context from previous tokens is essential. `use_cache=True` enables the model to store and reuse the key-value pairs from previous tokens. This caching mechanism is particularly important for maintaining the contextual continuity required for coherent and context-aware outputs.

Without `use_cache=True`, the model would treat each token in isolation, losing the accumulated context and significantly degrading the quality of the generation or prediction task, especially in long sequences.

In summary, `use_cache=True` is a directive that optimizes the model's ability to handle sequential token processing by preserving and leveraging the contextual information from the self-attention mechanism across the sequence. This leads to more efficient and contextually coherent outcomes in language modeling tasks.
"""
model.generate = partial(
    model.generate, language="english", task="transcribe", forced_decoder_ids=None, use_cache=True
)
```

**Load the Dataset**

We need to prepare the raw dataset for training where each transcript sentence matches its audio WhisperForConditionalGeneration

<u>Pre-processing</u>

We need to prepare the dataset in a way that is compatible with the model’s (encoder) input and (decoder) output.

General pre-processing steps:

1. Loading the audio file and its sampling rate.
2. Computing the log-Mel input features from the audio array.
3. Encoding the target text to label ids 1.

https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning

Whisper model has an associated feature extractor and tokenizer
- WhisperFeatureExtractor
- WhisperTokenizer 

These 2 objects are wrapped under a single class: WhisperProcessor

We need the feature extractor to pre-processes the raw audio-inputs to log-mel spectrograms

We need the tokenizer to post-processes the predicted tokens to text 

```
def prepare_dataset(batch):
    """
    Prepare the dataset for training.

    Args:
        batch (Dict[str, Any]): The batch to prepare.

    Returns:
        Dict[str, Any]: The prepared batch.
    """
    # load
    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = processer.tokenizer(batch["transcript"]).input_ids

    return batch
```

Or we can do

```
from transformers import WhisperFeatureExtractor, WhisperTokenizer


# generate log-Mel spectogram
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-small", language="English", task="transcribe"
)

# generate BPE tokens
tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="English", task="transcribe"
    )
```

Result is a DatasetDict

```
DatasetDict({
    train: Dataset({
        features: ['audio', 'transcript'],
        num_rows: 31728
    })
    validation: Dataset({
        features: ['audio', 'transcript'],
        num_rows: 3966
    })
    test: Dataset({
        features: ['audio', 'transcript'],
        num_rows: 3967
    })
})
```

**Training**

As we will be training in batches, it is important to ensure that the input and output features are of the same length. 

This is achieved by padding the input features and labels to the maximum length. 

In addition, we will also need to ensure each input is correctly paired with its corresponding output.

Custom data collator
- treats the input_features and labels independently: the input_features must be handled by the feature extractor and the labels by the tokenizer.

```
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for speech seq2seq with padding.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
```

- The output (labels) is padded to the maximum length, with padded tokens replaced by -100 to ensure that they are not factored into the loss computation.
- If you are fine-tuning on a dataset with audio samples longer than 30 seconds, you may need to chunk the audio accordingly.

Compute metrics:

```
import evaluate


metric = evaluate.load("wer")

def compute_metrics(pred) -> Dict[str, float]:
    """
    Compute metrics for the model.

    This function computes the Word Error Rate (WER) between the predicted and reference transcripts.
    The WER is a common metric used to evaluate the performance of automatic speech recognition (ASR) systems.
    It is calculated as the number of errors (insertions, deletions, and substitutions) divided by the total number of words in the reference transcript, 
    multiplied by 100 to get a percentage.

    Args:
        pred (transformers.EvalPrediction): The predictions and label_ids from the model.

    Returns:
        Dict[str, float]: A dictionary containing the WER metric.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)  # type: ignore

    return {"wer": wer}
```
- Since we replaced the padded tokens with -100 during the evaluation process, we will need to convert them back to the padding token (<|endoftext|>) before computing the metrics.

Initialise the training parameters and pass the 
- trainig arguements
- model
- dataset
- data collator
- compute metrics function

to the HuggingFace Trainer

```
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-singlish",
    per_device_train_batch_size=128,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    max_grad_norm=1.0,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vectorized_datasets["train"],
    eval_dataset=vectorized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
```

Training subset1:
- Training Batch size: 64
- 5000 steps
- 6-7 hours of training
- Divergence between training and eval loss signals potential overfitting
    - Training loss continually decreasing, evaluation loss plateaued
    - May indicate that model is [overly specialised](https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Deep-Dive-Into-Learning-Curves-in-Machine-Learning--Vmlldzo0NjA1ODY0) in fitting the training data 
    - Based on the author's research, general guideline is to fine-tune for no more than 2-4 epochs
    - Need to regularly checkpoint within this range to capture the best model
    - After evaluating loss curve, select checkpoint at 2000th step: Best balance where training loss is still relatively low and validation loss has not started to rise
    - The selection of the best model takes into account its performance on both training and validation data. A model that performs similarly well on both is more likely to generalize effectively to new, unseen data.

Training subset2, upon observations from subset1
- Training Batch size: 128
- 4000 steps
- Evaluation Batch size: 32
- Prematurely stopped training when evaluation loss plateaued and signs of overfitting were present
- 29-30 hours of training
- From around the 2,000th step onwards, a divergence between training and evaluation loss becomes noticeable, which suggests potential overfitting
- In general, we observe a similar training pattern for both subsets. However, with more data, the model demonstrates an improved ability to generalize to unseen data.
- Selected the checkpoint at the 3000th step as the final model 
- Conclusion: More data results in better fine-tuning performance

Other notes from the author
- whisper-large-v3 handles singlish well, better transcription then the fine-tuned ones 
- downside: higher costs and longer inference times 

Tips and other notes
- Transfer only the necessary data you need
- Implementing the EarlyStoppingCallback can help stop the training early if the model’s performance stops improving. Additionally, using an auto-pause feature for your instance can prevent incurring further costs when the instance is idle, saving both time and money.
- If training loss continually decreases but eval loss plateaus, model may be [overly specialised](https://wandb.ai/mostafaibrahim17/ml-articles/reports/A-Deep-Dive-Into-Learning-Curves-in-Machine-Learning--Vmlldzo0NjA1ODY0) in fitting the training data
- To fine-tune a more robust model, audio samples should include variations in speech formats 
- Explore PEFT methods
- Explore [distilled](https://github.com/huggingface/distil-whisper) Whisper Model: Much smaller and faster
- 

**Results**

- base-whisper-small: 57.9%
- whisper-small-singlish-39k: 13.59% (subset1)
- whisper-small-singlish-122k: 9.69% (subset2)

