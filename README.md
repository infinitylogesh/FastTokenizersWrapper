# Fast Tokenizers Wrapper

A wrapper for Huggingface's [Tokenizers](https://github.com/huggingface/tokenizers) library , for it to be used along with existing version of Huggingface's Transformers.Tokenizers from Tokenizers library,  are much faster compared to Transformers' native tokenizers. This wrapper ```FastTokenizers.py``` can be used along with existing version of transformers library.

```BertTokenizerFast``` and ```DistilBertTokenizerFast``` are the wrappers for bert and distilBert tokenizers using [tokenizers](https://github.com/huggingface/tokenizers) library.

## Usage :

Usage is very similar to [BertTokenizer](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer) and [DistilBertTokenizer](https://huggingface.co/transformers/model_doc/distilbert.html#distilberttokenizer) class in transformers library.

```python

from FastTokenizers import DistilBertTokenizerFast,BertTokenizerFast

# Tokenizer can be initialized without a vocab file as in Transformers library.
fastDistilTokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased',
                                                               do_lower_case=True,
                                                               cache_dir=None)

fastBertTokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',
                                                      do_lower_case=True,
                                                      cache_dir=None)

```

## Language Model Finetuning with Fast Tokenizers Wrapper :

LM finetuning is much faster with tokenizers, `run_lm_finetuning.py` script is updated with FastTokenizers. Invoking and usage of the script is as same as the original script on [Huggingface's Transformers](https://huggingface.co/transformers/examples.html#language-model-fine-tuning)

## Credits

The scripts were adapted from Huggingface's [Transformers](https://huggingface.co/transformers) library.Inspired from yet to be released, Huggingface's BertTokenizerFast.
