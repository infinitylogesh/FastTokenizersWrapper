from transformers import PreTrainedTokenizer
from transformers.tokenization_utils import is_tf_available,is_torch_available
from transformers.tokenization_bert import VOCAB_FILES_NAMES,PRETRAINED_VOCAB_FILES_MAP,PRETRAINED_INIT_CONFIGURATION,PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
from transformers import tokenization_distilbert as tdb
import logging
logger = logging.getLogger(__name__)
import six
if is_tf_available():
    import tensorflow as tf
if is_torch_available():
    import torch

class FastPreTrainedTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super(FastPreTrainedTokenizer, self).__init__(**kwargs)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise NotImplementedError
        return self._tokenizer

    @property
    def decoder(self):
        if self._decoder is None:
            raise NotImplementedError
        return self._decoder

    @property
    def vocab_size(self):
        return self.tokenizer._tokenizer.get_vocab_size()

    def __len__(self):
        return self.tokenizer._tokenizer.get_vocab_size()

    def _update_special_tokens(self):
        self.tokenizer.add_special_tokens(self.all_special_tokens)

    @staticmethod
    def _convert_encoding(encoding,
                          return_tensors=None,
                          return_token_type_ids=True,
                          return_attention_mask=True,
                          return_overflowing_tokens=False,
                          return_special_tokens_mask=False):
        encoding_dict = {
            "input_ids": encoding.ids,
        }
        if return_token_type_ids:
            encoding_dict["token_type_ids"] = encoding.type_ids
        if return_attention_mask:
            encoding_dict["attention_mask"] = encoding.attention_mask
        if return_overflowing_tokens:
            overflowing = encoding.overflowing
            encoding_dict["overflowing_tokens"] = overflowing.ids if overflowing is not None else []
        if return_special_tokens_mask:
            encoding_dict["special_tokens_mask"] = encoding.special_tokens_mask

        # Prepare inputs as tensors if asked
        if return_tensors == 'tf' and is_tf_available():
            encoding_dict["input_ids"] = tf.constant([encoding_dict["input_ids"]])
            encoding_dict["token_type_ids"] = tf.constant([encoding_dict["token_type_ids"]])

            if "attention_mask" in encoding_dict:
                encoding_dict["attention_mask"] = tf.constant([encoding_dict["attention_mask"]])

        elif return_tensors == 'pt' and is_torch_available():
            encoding_dict["input_ids"] = torch.tensor([encoding_dict["input_ids"]])
            encoding_dict["token_type_ids"] = torch.tensor([encoding_dict["token_type_ids"]])

            if "attention_mask" in encoding_dict:
                encoding_dict["attention_mask"] = torch.tensor([encoding_dict["attention_mask"]])
        elif return_tensors is not None:
            logger.warning(
                "Unable to convert output to tensors format {}, PyTorch or TensorFlow is not available.".format(
                    return_tensors))

        return encoding_dict

    def encode_plus(self,
                    text,
                    text_pair=None,
                    return_tensors=None,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    return_overflowing_tokens=False,
                    return_special_tokens_mask=False,
                    **kwargs):
        encoding = self.tokenizer.encode(text, text_pair)
        return self._convert_encoding(encoding,
                                      return_tensors=return_tensors,
                                      return_token_type_ids=return_token_type_ids,
                                      return_attention_mask=return_attention_mask,
                                      return_overflowing_tokens=return_overflowing_tokens,
                                      return_special_tokens_mask=return_special_tokens_mask)

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

    def _convert_token_to_id_with_added_voc(self, token):
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index):
        return self.tokenizer.id_to_token(int(index))

    def convert_tokens_to_string(self, tokens):
        return self.decoder.decode(tokens)

    def add_tokens(self, new_tokens):
        self.tokenizer.add_tokens(new_tokens)

    def encode_batch(self, texts,
                     return_tensors=None,
                     return_token_type_ids=True,
                     return_attention_mask=True,
                     return_overflowing_tokens=False,
                     return_special_tokens_mask=False):
        return [self._convert_encoding(encoding,
                                       return_tensors=return_tensors,
                                       return_token_type_ids=return_token_type_ids,
                                       return_attention_mask=return_attention_mask,
                                       return_overflowing_tokens=return_overflowing_tokens,
                                       return_special_tokens_mask=return_special_tokens_mask)
                for encoding in self.tokenizer.encode_batch(texts)]

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        text = self.tokenizer.decode(token_ids, skip_special_tokens)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def decode_batch(self, ids_batch, skip_special_tokens=False, clear_up_tokenization_spaces=True):
        return [self.clean_up_tokenization(text)
                if clear_up_tokenization_spaces else text
                for text in self.tokenizer.decode_batch(ids_batch, skip_special_tokens)]

class BertTokenizerFast(FastPreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=True,
                 max_length=None, pad_to_max_length=False, stride=0,
                 truncation_strategy='longest_first', add_special_tokens=True, **kwargs):

        try:
            from tokenizers import BertWordPieceTokenizer
            super(BertTokenizerFast, self).__init__(unk_token=unk_token, sep_token=sep_token,
                                                    pad_token=pad_token, cls_token=cls_token,
                                                    mask_token=mask_token, **kwargs)

            self._tokenizer = BertWordPieceTokenizer(
                vocab_file,
                unk_token=unk_token,
                sep_token=sep_token,
                cls_token=cls_token,
                handle_chinese_chars=tokenize_chinese_chars,
                lowercase=do_lower_case,
                add_special_tokens=add_special_tokens
            )
            self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
            self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens
            self._update_special_tokens()
            
            if max_length is not None:
                self._tokenizer.with_truncation(max_length, stride, truncation_strategy)

        except (AttributeError, ImportError) as e:
            logger.error("Make sure you installed `tokenizers` with `pip install tokenizers==0.2.1`")
            raise e

class DistilBertTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a DistilBertTokenizer.
    :class:`~transformers.DistilBertTokenizer` is identical to BertTokenizer and runs end-to-end tokenization: punctuation splitting + wordpiece

    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_wordpiece_only=False
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying BERT model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_wordpiece_only=False
    """

    vocab_files_names = tdb.VOCAB_FILES_NAMES
    pretrained_vocab_files_map = tdb.PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = tdb.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

