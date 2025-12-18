import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import torch
import json
import os
from models.albert_model import ALBERTForNER, ALBERTNERTrainer
from utils.preprocess import DataPreprocessor
from utils.metrics import MetricsCalculator
from utils.visualization import EntityVisualizer


class LegalNERGUI:
    def __init__(self, root, model, preprocessor, trainer):
        self.root = root
        self.model = model
        self.preprocessor = preprocessor
        self.trainer = trainer
        self.visualizer = EntityVisualizer()

        self.root.title("ALBERT Legal NER System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')

        self.create_widgets()

    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#34495e', height=80)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title = tk.Label(
            title_frame,
            text="⚖️ ALBERT Legal Named Entity Recognition",
            font=('Arial', 20, 'bold'),
            bg='#34495e',
            fg='white'
        )
        title.pack(pady=20)

        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Input
        left_panel = tk.Frame(main_container, bg='#34495e', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        input_label = tk.Label(
            left_panel,
            text="📝 Input Legal Text",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        )
        input_label.pack(pady=10)

        self.input_text = scrolledtext.ScrolledText(
            left_panel,
            height=15,
            font=('Arial', 11),
            wrap=tk.WORD,
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        self.input_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Analyze button
        analyze_btn = tk.Button(
            left_panel,
            text="🔍 Analyze Text",
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            command=self.analyze_text,
            cursor='hand2',
            relief=tk.RAISED,
            borderwidth=3
        )
        analyze_btn.pack(pady=10)

        # Right panel - Output
        right_panel = tk.Frame(main_container, bg='#34495e', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        output_label = tk.Label(
            right_panel,
            text="✨ Annotated Output",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        )
        output_label.pack(pady=10)

        self.output_text = scrolledtext.ScrolledText(
            right_panel,
            height=15,
            font=('Arial', 11),
            wrap=tk.WORD,
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        self.output_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Legend
        self.visualizer.create_legend(right_panel)

        # Bottom panel - Metrics and Chart
        bottom_panel = tk.Frame(self.root, bg='#2c3e50')
        bottom_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Metrics panel
        metrics_panel = tk.Frame(bottom_panel, bg='#34495e', relief=tk.RAISED, borderwidth=2)
        metrics_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        metrics_label = tk.Label(
            metrics_panel,
            text="📊 Model Metrics",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        )
        metrics_label.pack(pady=10)

        self.metrics_text = scrolledtext.ScrolledText(
            metrics_panel,
            height=10,
            font=('Courier', 10),
            wrap=tk.WORD,
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        self.metrics_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Chart panel
        self.chart_panel = tk.Frame(bottom_panel, bg='#34495e', relief=tk.RAISED, borderwidth=2)
        self.chart_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        chart_label = tk.Label(
            self.chart_panel,
            text="📈 Entity Distribution",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        )
        chart_label.pack(pady=10)

    def analyze_text(self):
        """Analyze input text and display results"""
        input_text = self.input_text.get('1.0', tk.END).strip()

        if not input_text:
            messagebox.showwarning("Warning", "Please enter some text to analyze!")
            return

        # Tokenize input
        words = input_text.split()
        text = " ".join(words)

        encoding = self.preprocessor.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.preprocessor.max_length,
            return_tensors='pt'
        )

        # Predict
        inputs = [{
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }]

        predictions = self.trainer.predict(inputs)
        pred_tags = [self.preprocessor.idx2tag[p] for p in predictions[0]]

        # Extract entities
        word_ids = encoding.word_ids(batch_index=0)
        word_tags = []
        previous_word_idx = None

        for word_idx, tag_idx in zip(word_ids, predictions[0]):
            if word_idx is not None and word_idx != previous_word_idx:
                if word_idx < len(words):
                    word_tags.append((words[word_idx], self.preprocessor.idx2tag[tag_idx]))
                previous_word_idx = word_idx

        # Extract entities for visualization
        entity_words = [w[0] for w in word_tags]
        entity_tags = [w[1] for w in word_tags]
        entities = self.visualizer.extract_entities(entity_words, entity_tags)

        # Display highlighted output
        self.visualizer.highlight_entities(self.output_text, text, entities)

        # Update chart
        self.visualizer.create_entity_chart(self.chart_panel, entities)

        # Save results
        self.save_results(text, entities)

        messagebox.showinfo("Success", f"Analysis complete! Found {len(entities)} entities.")

    def save_results(self, text, entities):
        """Save results to JSON and text files"""
        os.makedirs('outputs', exist_ok=True)

        # JSON output
        results = {
            'text': text,
            'entities': entities
        }

        with open('outputs/results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Annotated text output
        annotated = text
        for entity in sorted(entities, key=lambda x: len(x['text']), reverse=True):
            annotated = annotated.replace(
                entity['text'],
                f"[{entity['type']}: {entity['text']}]"
            )

        with open('outputs/annotated_output.txt', 'w', encoding='utf-8') as f:
            f.write(annotated)

    def display_metrics(self, metrics_text):
        """Display metrics in the metrics panel"""
        self.metrics_text.delete('1.0', tk.END)
        self.metrics_text.insert('1.0', metrics_text)


def main():
    print("=" * 60)
    print("ALBERT Legal NER System - Initializing...")
    print("=" * 60)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize preprocessor
    print("\n[1/6] Initializing preprocessor...")
    preprocessor = DataPreprocessor(max_length=128)

    # Load and prepare data
    print("[2/6] Loading training data...")
    train_inputs, train_labels, train_sents, train_tags = preprocessor.prepare_data('data/train.txt')

    print("[3/6] Loading test data...")
    test_inputs, test_labels, test_sents, test_tags = preprocessor.prepare_data('data/test.txt')

    # Initialize model
    print("[4/6] Initializing ALBERT model...")
    num_labels = len(preprocessor.tag2idx)
    model = ALBERTForNER(num_labels=num_labels)
    trainer = ALBERTNERTrainer(model, device=device)

    # Train model
    print("[5/6] Training model...")
    trainer.train(train_inputs, train_labels, epochs=5, batch_size=16)

    # Evaluate model
    print("[6/6] Evaluating model...")
    predictions = trainer.predict(test_inputs)

    # Calculate metrics
    metrics_calc = MetricsCalculator(preprocessor.idx2tag)
    metrics = metrics_calc.calculate_metrics(
        [label.tolist() for label in test_labels],
        predictions
    )
    entity_metrics = metrics_calc.calculate_entity_metrics(
        [label.tolist() for label in test_labels],
        predictions
    )

    metrics_text = metrics_calc.format_metrics(metrics, entity_metrics)
    print(metrics_text)

    # Save model
    print("\nSaving model...")
    trainer.save_model('outputs/albert_ner_model')

    # Launch GUI
    print("\nLaunching GUI...")
    root = tk.Tk()
    app = LegalNERGUI(root, model, preprocessor, trainer)
    app.display_metrics(metrics_text)

    # Set sample text
    sample_text = "Supreme Court of India delivered judgment on 12th July 2024. Article 370 was abrogated. Justice Ravi Menon presided over the case."
    app.input_text.insert('1.0', sample_text)

    print("\n" + "=" * 60)
    print("GUI Launched Successfully!")
    print("=" * 60)

    root.mainloop()


if __name__ == "__main__":
    main()