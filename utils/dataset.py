import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, RobertaTokenizerFast
from utils.utils import match_labels
import random

class LegalNERTokenDataset(Dataset):
    
    def __init__(self, dataset_path, model_path, labels_list=None, split="train", use_roberta=False, augment=True, original_label_list=["JUDGE","COURT"]):
        self.data = json.load(open(dataset_path))
        self.split = split
        self.use_roberta = use_roberta
        self.original_label_list = original_label_list or []

        if self.use_roberta:
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.labels_list = sorted(labels_list + ["O"])[::-1]
        self.augment = augment

        if self.labels_list is not None:
            self.labels_to_idx = dict(
                zip(sorted(self.labels_list)[::-1], range(len(self.labels_list)))
            )

        if self.augment:
            augmented_data = []
            for item in self.data:
                original_text = item["data"]["text"]
                annotations = [
                    {
                        "start": v["value"]["start"],
                        "end": v["value"]["end"],
                        "labels": v["value"]["labels"][0],
                    }
                    for v in item["annotations"][0]["result"]
                ]

                augmented_text = self.mention_replacement_augmentation(original_text, annotations)

                if augmented_text != original_text:
                    augmented_item = {
                        "data": {"text": augmented_text},
                        "annotations": item["annotations"]
                    }
                    augmented_data.append(augmented_item)

            self.data.extend(augmented_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if 'data' not in item or 'text' not in item['data']:
            print(f"Warning: 'data' key not found in dataset item at index {idx}.")
            return None

        text = item["data"]["text"]
        annotations = [
            {
                "start": v["value"]["start"],
                "end": v["value"]["end"],
                "labels": v["value"]["labels"][0],
            }
            for v in item["annotations"][0]["result"]
        ]

        inputs = self.tokenize_text(text)
        aligned_labels = match_labels(inputs, annotations)
        aligned_labels = [self.labels_to_idx[l] for l in aligned_labels]
        self.apply_labels_to_inputs(inputs, aligned_labels)

        return inputs

    def tokenize_text(self, text):
        if not self.use_roberta:
            return self.tokenizer(text, return_tensors="pt", truncation=True, verbose=False)
        else:
            return self.tokenizer(text, return_tensors="pt", truncation=True, verbose=False, padding='max_length')

    def apply_labels_to_inputs(self, inputs, labels):
        inputs["input_ids"] = inputs["input_ids"].squeeze(0).long()
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0).long()
        if not self.use_roberta:
            inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0).long()
        labels = torch.tensor(labels).squeeze(-1).long()
        if labels.shape[0] < inputs["attention_mask"].shape[0]:
            pad_x = torch.zeros((inputs["input_ids"].shape[0],))
            pad_x[: labels.size(0)] = labels
            inputs["labels"] = labels
        else:
            inputs["labels"] = labels[: inputs["attention_mask"].shape[0]]

    def mention_replacement_augmentation(self, text, annotations):
        entities = [ann for ann in annotations if ann['labels'] in self.original_label_list]
        entity_pool = {label: [] for label in self.original_label_list}
        for entity in entities:
            entity_text = text[entity['start']:entity['end']]
            entity_pool[entity['labels']].append(entity_text)

        augmented_text = text
        for entity in entities:
            entity_label = entity['labels']
            matching_entities = [e for e in entity_pool[entity_label] if len(e) == len(text[entity['start']:entity['end']])]
            if matching_entities:
                replacement_entity = random.choice(matching_entities)
                start, end = entity['start'], entity['end']
                augmented_text = augmented_text[:start] + replacement_entity + augmented_text[end:]

        return augmented_text


