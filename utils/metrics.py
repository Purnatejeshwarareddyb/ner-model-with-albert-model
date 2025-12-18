import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict


class MetricsCalculator:
    def __init__(self, idx2tag):
        self.idx2tag = idx2tag

    def calculate_metrics(self, true_labels, pred_labels):
        """Calculate overall metrics"""
        # Flatten and filter out PAD tokens
        true_flat = []
        pred_flat = []

        for true_seq, pred_seq in zip(true_labels, pred_labels):
            for t, p in zip(true_seq, pred_seq):
                if self.idx2tag[t] != 'PAD':
                    true_flat.append(t)
                    pred_flat.append(p)

        # Calculate metrics
        accuracy = accuracy_score(true_flat, pred_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_flat, pred_flat, average='weighted', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def calculate_entity_metrics(self, true_labels, pred_labels):
        """Calculate per-entity metrics"""
        entity_types = ['LAW', 'CASE', 'DATE', 'ORG', 'PERSON']
        entity_metrics = {}

        for entity in entity_types:
            true_binary = []
            pred_binary = []

            for true_seq, pred_seq in zip(true_labels, pred_labels):
                for t, p in zip(true_seq, pred_seq):
                    true_tag = self.idx2tag[t]
                    pred_tag = self.idx2tag[p]

                    if true_tag != 'PAD':
                        # Check if entity type matches
                        true_is_entity = entity in true_tag
                        pred_is_entity = entity in pred_tag
                        true_binary.append(int(true_is_entity))
                        pred_binary.append(int(pred_is_entity))

            if sum(true_binary) > 0:  # Only calculate if entity exists
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_binary, pred_binary, average='binary', zero_division=0
                )
                entity_metrics[entity] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
            else:
                entity_metrics[entity] = {
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1': 1.0
                }

        return entity_metrics

    def format_metrics(self, metrics, entity_metrics):
        """Format metrics for display"""
        output = "\n" + "=" * 50 + "\n"
        output += "Model: ALBERT for Legal NER\n"
        output += "=" * 50 + "\n"
        output += f"Accuracy:  {metrics['accuracy'] * 100:.2f}%\n"
        output += f"Precision: {metrics['precision']:.2f}\n"
        output += f"Recall:    {metrics['recall']:.2f}\n"
        output += f"F1-Score:  {metrics['f1']:.2f}\n"
        output += "=" * 50 + "\n"
        output += "Per-Entity Results:\n"
        output += f"{'Entity':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}\n"
        output += "-" * 50 + "\n"

        for entity, scores in entity_metrics.items():
            output += f"{entity:<10} {scores['precision']:<12.2f} {scores['recall']:<12.2f} {scores['f1']:<12.2f}\n"

        output += "=" * 50 + "\n"
        return output