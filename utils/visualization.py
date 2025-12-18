import tkinter as tk
from tkinter import scrolledtext, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter


class EntityVisualizer:
    def __init__(self):
        self.entity_colors = {
            'LAW': '#3498db',  # Blue
            'PERSON': '#2ecc71',  # Green
            'ORG': '#f1c40f',  # Yellow
            'DATE': '#e67e22',  # Orange
            'CASE': '#9b59b6',  # Purple
            'O': '#ecf0f1'  # Light gray
        }

    def extract_entities(self, words, tags):
        """Extract entities from words and tags"""
        entities = []
        current_entity = []
        current_type = None

        for word, tag in zip(words, tags):
            if tag.startswith('B-'):
                if current_entity:
                    entities.append({
                        'text': ' '.join(current_entity),
                        'type': current_type
                    })
                current_entity = [word]
                current_type = tag[2:]
            elif tag.startswith('I-') and current_entity:
                current_entity.append(word)
            else:
                if current_entity:
                    entities.append({
                        'text': ' '.join(current_entity),
                        'type': current_type
                    })
                    current_entity = []
                    current_type = None

        if current_entity:
            entities.append({
                'text': ' '.join(current_entity),
                'type': current_type
            })

        return entities

    def highlight_entities(self, text_widget, text, entities):
        """Highlight entities in text widget with colors"""
        text_widget.delete('1.0', tk.END)
        text_widget.insert('1.0', text)

        # Configure tags for each entity type
        for entity_type, color in self.entity_colors.items():
            text_widget.tag_config(
                entity_type,
                background=color,
                foreground='white' if entity_type != 'O' else 'black',
                font=('Arial', 10, 'bold')
            )

        # Apply highlights
        for entity in entities:
            start_idx = '1.0'
            while True:
                start_idx = text_widget.search(
                    entity['text'],
                    start_idx,
                    stopindex=tk.END
                )
                if not start_idx:
                    break
                end_idx = f"{start_idx}+{len(entity['text'])}c"
                text_widget.tag_add(entity['type'], start_idx, end_idx)
                start_idx = end_idx

    def create_entity_chart(self, frame, entities):
        """Create a bar chart of entity distribution"""
        # Clear existing widgets
        for widget in frame.winfo_children():
            widget.destroy()

        # Count entities
        entity_counts = Counter([e['type'] for e in entities])

        if not entity_counts:
            label = tk.Label(frame, text="No entities detected", font=('Arial', 12))
            label.pack(pady=20)
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        types = list(entity_counts.keys())
        counts = list(entity_counts.values())
        colors = [self.entity_colors.get(t, '#95a5a6') for t in types]

        ax.bar(types, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Entity Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Entity Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_legend(self, frame):
        """Create a legend for entity colors"""
        legend_frame = tk.Frame(frame, bg='white', relief=tk.RIDGE, borderwidth=2)
        legend_frame.pack(pady=10, padx=10, fill=tk.X)

        title = tk.Label(
            legend_frame,
            text="Entity Color Legend",
            font=('Arial', 12, 'bold'),
            bg='white'
        )
        title.pack(pady=5)

        for entity_type, color in self.entity_colors.items():
            if entity_type != 'O':
                row = tk.Frame(legend_frame, bg='white')
                row.pack(fill=tk.X, padx=5, pady=2)

                color_box = tk.Label(
                    row,
                    text="  ",
                    bg=color,
                    width=3,
                    relief=tk.RAISED
                )
                color_box.pack(side=tk.LEFT, padx=5)

                label = tk.Label(
                    row,
                    text=entity_type,
                    font=('Arial', 10),
                    bg='white'
                )
                label.pack(side=tk.LEFT)