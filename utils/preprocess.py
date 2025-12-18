import os
from transformers import AlbertTokenizerFast

import torch


class DataPreprocessor:
    def __init__(self, max_length=128):
        self.tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
        self.max_length = max_length
        self.tag2idx = {
            'O': 0,
            'B-LAW': 1, 'I-LAW': 2,
            'B-CASE': 3, 'I-CASE': 4,
            'B-DATE': 5, 'I-DATE': 6,
            'B-ORG': 7, 'I-ORG': 8,
            'B-PERSON': 9, 'I-PERSON': 10,
            'PAD': 11
        }
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

    def read_data(self, file_path):
        """Read IOB formatted data"""
        if not os.path.exists(file_path):
            return [], []

        sentences = []
        labels = []
        current_sentence = []
        current_labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0]
                        tag = parts[1]
                        current_sentence.append(word)
                        current_labels.append(tag)
                else:
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []

        # Add last sentence if file doesn't end with blank line
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)

        return sentences, labels

    def tokenize_and_align_labels(self, sentences, labels):
        """Tokenize sentences and align labels with subword tokens"""
        tokenized_inputs = []
        aligned_labels = []

        for sentence, label in zip(sentences, labels):
            # Join words to form complete sentence
            text = " ".join(sentence)

            # Tokenize
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                is_split_into_words=False
            )

            # Create label alignment
            word_ids = encoding.word_ids(batch_index=0)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(self.tag2idx['PAD'])
                elif word_idx != previous_word_idx:
                    if word_idx < len(label):
                        label_ids.append(self.tag2idx[label[word_idx]])
                    else:
                        label_ids.append(self.tag2idx['O'])
                else:
                    if word_idx < len(label):
                        # For subword tokens, use I- tag or O
                        current_tag = label[word_idx]
                        if current_tag.startswith('B-'):
                            label_ids.append(self.tag2idx['I-' + current_tag[2:]])
                        else:
                            label_ids.append(self.tag2idx[current_tag])
                    else:
                        label_ids.append(self.tag2idx['O'])
                previous_word_idx = word_idx

            tokenized_inputs.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            })
            aligned_labels.append(torch.tensor(label_ids))

        return tokenized_inputs, aligned_labels

    def prepare_data(self, file_path):
        """Complete data preparation pipeline"""
        sentences, labels = self.read_data(file_path)
        tokenized_inputs, aligned_labels = self.tokenize_and_align_labels(sentences, labels)
        return tokenized_inputs, aligned_labels, sentences, labels