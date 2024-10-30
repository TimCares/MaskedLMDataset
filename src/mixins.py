"""
This module provides a text mixin that can be used to add text support to a dataset.
"""
import torch
from typing import *
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BertTokenizer
import random
from .utils import check_int_str_list_type
    
class TextMixin:
    def __init__(
        self,
        tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
        max_seq_len:int=512,
        mlm_probability:float=0.0,
        *args:List[Any],
        **kwargs:Dict[str, Any],
    ):
        """A Mixin designed to add text/tokenization and text masking support to a dataset.

        Args:
            tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast], optional): A tokenizer class that implements
                the Huggingface tokenizer API. Used to tokenize text data. Defaults to None.
                If None, then the BERT base uncased tokenizer will be used by default:
                BertTokenizer.from_pretrained("bert-base-uncased").
            max_seq_len (int, optional): The maximum sequence length of the tokenized text data. Defaults to 512.
            mlm_probability (float, optional): The probability of masking a token in the text data. Defaults to 0.0, so no masking is done.
            *args (List[Any]): Positional arguments passed to other Mixins, if present. Defaults to None.
            **kwargs (Dict[str, Any]): Keyword arguments passed to other Mixins, if present. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.tokenizer_path = ".tokenizers"
        self.tokenizer:PreTrainedTokenizer = BertTokenizer.from_pretrained("bert-base-uncased") if tokenizer is None else tokenizer
        self.tokenizer.save_pretrained(self.tokenizer_path)
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability

    @property
    def cls_token_id(self) -> int:
        return self.tokenizer.cls_token_id
    
    @property
    def sep_token_id(self) -> int:
        return self.tokenizer.sep_token_id
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_text(self, text:Union[str, List[str], List[int]], whole_word_mask:bool=True) -> Dict[str, List[int]]:
        """Tokenizes the text, masks tokens (optional), and pads it to the maximum sequence length.

        Args:
            text (Union[str, List[str], List[int]]): The text. Either already tokenized as a list of subwords,
                tokenized as a list of token ids, or the raw text as a string.
            whole_word_mask (bool, optional): Whether to mask whole words or individual tokens.
                Ignored if mlm_probability == 0. Defaults to True.

        Returns:
            Dict[str, List[int]]: A dictionary containing the tokenized text (key "text"),
                the padding mask (key "padding_mask"), and the mask labels
                (key "mask_labels", only present if self.mlm_probability > 0).
        """
        tokens = self.tokenize_text(text)
        assert len(tokens) > 0, "The text should contain at least one token!"
        
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2] # truncate the tokens to the maximum sequence length, -2 because of [CLS] and [SEP]
        mask_labels = self.mask_sequence(input_tokens=tokens, whole_word_mask=whole_word_mask)

        tokens = [self.cls_token_id] + tokens + [self.sep_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (self.max_seq_len - num_tokens)
        language_tokens = tokens + [self.pad_token_id] * (self.max_seq_len - num_tokens)

        result_dict = {
            "text": language_tokens,
            "padding_mask": padding_mask,
        }

        if mask_labels is not None:
            # pad the mask labels to the maximum sequence length
            # first token will be [CLS], that is why we prepend [0]
            # after last token, there will be a [SEP] and the rest will be padded, both padded tokens and [SEP] will never be masked,
            # that is why we append as many 0 as needed until the maximum sequence length is reached
            mask_labels = [0] + mask_labels + [0] * (self.max_seq_len - len(mask_labels) - 1)
            result_dict["mask_labels"] = mask_labels
        
        return result_dict

    def tokenize_text(self, text:Union[str, List[str], List[int]]) -> Union[int, List[int]]:
        """Wrapper for easy tokenization of text data. Converts text to tokens and then to token ids using the
        (HuggingFace) tokenizer provided during initialization. Tokens are not truncated or padded, and do not
        include special tokens like [CLS] or [SEP].

        Args:
            text (Union[str, List[str], List[int]]): The text. Either already tokenized as a list of subwords,
                tokenized as a list of token ids, or the raw text as a string.
                If text is already tokenized and converted to token ids, then this method just returns the input.

        Returns:
            Union[int, List[int]]: Token ids of the text, without special tokens.
        """
        assert isinstance(text, (str, list)), "The text should be either a string or a list of strings or list of integers!"
        if isinstance(text, str):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text.strip()))
        list_type = check_int_str_list_type(text)
        if list_type == str:
            return self.tokenizer.convert_tokens_to_ids(text)
        if list_type == Any:
            raise ValueError("The text should be either a string or a list of strings or list of integers! Found mixed types.")
        
        return text # text is already tokenized and converted to token ids -> List[int] (check_int_str_list_type returned int)
    
    def mask_sequence(self, input_tokens: List[int], whole_word_mask:bool=True) -> Union[List[int], None]:
        """Mask tokens in the input sequence either completely independent for each other,
        or one whole words, so subwords together. Special tokens [CLS] and [SEP] are not masked.

        Args:
            input_tokens (List[int]): The input sequence of tokens, already tokenized and converted to token ids.
            whole_word_mask (bool, optional): Whether to mask whole words, setting to True is encouraged. Defaults to True.

        Returns:
            Union[List[int], None]: if self.mlm_probability > 0, then: The mask indicator for each token in the
                input sequence. 1 is mask, 0 is not masked.
                If self.mlm_probability == 0, then None is returned.
        """        
        if self.mlm_probability == 0.0: # if no masking is required, return a list of zeros because no tokens are masked
            return None
        if whole_word_mask:
            # sequence is already tokenized and converted to token ids, to detect which token is a subword,
            # we need to convert the tokens ids back to tokens (tokens that are subwords start with "##")
            input_tokens:List[str] = self.tokenizer.convert_ids_to_tokens(input_tokens)
            return self._whole_word_mask(input_tokens)
        return self._token_mask(input_tokens)

    def _token_mask(self, input_tokens: List[int]) -> List[int]:
        """Mask tokens in the input sequence with the given probability. Special tokens [CLS] and [SEP] are not masked.
        Sequence is expected to not be padded.

        Args:
            input_tokens (List[int]): The input sequence of tokens, already tokenized and converted to token ids.

        Returns:
            List[int]: The mask indicator for each token in the input sequence. 1 is mask, 0 is not masked.
        """        
        input_tokens = torch.Tensor(input_tokens)
        special_tokens_mask = input_tokens == self.sep_token_id or input_tokens == self.cls_token_id
        probabilities = torch.full(input_tokens.shape, self.mlm_probability)
        probabilities.masked_fill_(special_tokens_mask, value=0.0)
        return torch.bernoulli(probabilities).tolist()

    # adjusted from transformers.data.data_collator.DataCollatorForWholeWordMask
    def _whole_word_mask(self, input_tokens: List[str]) -> List[int]:
        """Performs whole word masking on a sequence.
        If we have a word that is split into multiple subwords e.g. "bicycle" -> ["bi", "##cycle"],
        and the first subword "bi" is masked, then the second subword "##cycle" will also be masked.
        Special tokens [CLS] and [SEP] are not masked.
        Sequence is expected to not be padded.

        Args:
            input_tokens (List[int]): The input sequence of tokens, has to be the raw tokens, not token ids.
                Subwords must start with "##".

        Returns:
            List[int]: The mask indicator for each token in the input sequence. 1 is mask, 0 is not masked.
        """         
        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = max(1, int(round(len(input_tokens) * self.mlm_probability)))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    # adjusted from transformers.data.data_collator.DataCollatorForWholeWordMask
    def apply_mask(self, batch:Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply the masking to the batched text. Of all masked tokens, 80% are replaced by the mask token,
        10% are replaced by a random token, and 10% are left unchanged. The mask labels are used to indicate

        Args:
            batch (Dict[str, torch.Tensor]): The batch dictionary containing the input tokens (key "text"), and
                optionally the mask labels (key "mask_labels"), which indicate which tokens should be masked (1) and which not (0).
                Both "text" and "mask_labels" are expected to be tensors of shape (batch_size, max_seq_len).

        Returns:
            Dict[str, torch.Tensor]: A batch dictionary with the (now) masked input tokens (key "text"), and the labels (key "labels").
                If no masking is done (self.mlm_probability == 0 or "mask_labels" not in batch), then the input batch is returned unchanged.
        """
        assert 'text' in batch, "The batch dictionary should contain the key 'text'!"
        if self.mlm_probability == 0.0 or 'mask_labels' not in batch:
            return batch
        
        input_tokens = batch['text']
        labels = input_tokens.clone()

        masked_indices = batch['mask_labels'].bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_tokens[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_tokens[indices_random] = random_words[indices_random]

        # ... the rest of the time (10% of the time) we keep the masked input tokens unchanged

        batch['text'] = input_tokens
        batch['labels'] = labels
        return batch
