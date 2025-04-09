import sys
import pandas as pd
import numpy as np
import torch
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal
from sentence_transformers import SentenceTransformer
from collections import Counter


class RecommendationWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, embedding_model, look_embeddings, product_map, look_product_map):
        super().__init__()
        self.embedding_model = embedding_model
        self.look_embeddings = look_embeddings
        self.product_map = product_map
        self.look_product_map = look_product_map
        self.query = ""

    def run(self):
        try:
            # Encode user query
            query_embedding = self.embedding_model.encode(self.query)
            query_tensor = torch.FloatTensor(query_embedding)

            # Convert look embeddings to tensor
            look_embeddings_tensor = torch.FloatTensor(self.look_embeddings)

            # Calculate cosine similarities
            similarities = torch.cosine_similarity(
                query_tensor.unsqueeze(0),
                look_embeddings_tensor,
                dim=1
            )

            # Get top 5 indices
            _, indices = torch.topk(similarities, 5)
            indices = indices.cpu().numpy()

            # Collect recommended products
            recommended_products = []
            for idx in indices:
                recommended_products.extend(self.look_product_map[idx])

            # Get top 10 most common products
            product_counts = Counter(recommended_products)
            top_products = [self.product_map[pid] for pid, _ in product_counts.most_common(10)]

            self.finished.emit(top_products)

        except Exception as e:
            self.error.emit(f"Recommendation error: {str(e)}")


class FashionRecommender(QWidget):
    def __init__(self):
        super().__init__()
        self.embedding_model = None
        self.look_embeddings = None
        self.product_map = {}
        self.look_product_map = {}
        self.init_ui()
        self.load_data()

    def init_ui(self):
        self.setWindowTitle('Fashion Recommendation System')
        self.setGeometry(300, 300, 600, 400)

        layout = QVBoxLayout()

        # Input widgets
        self.input_label = QLabel("Enter your style request:")
        self.input_field = QLineEdit()
        self.submit_btn = QPushButton("Get Recommendations")
        self.submit_btn.clicked.connect(self.handle_recommendation)

        # Results display
        self.results_area = QTextEdit()
        self.results_area.setReadOnly(True)

        # Add widgets to layout
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_field)
        layout.addWidget(self.submit_btn)
        layout.addWidget(self.results_area)

        self.setLayout(layout)

    def load_data(self):
        try:
            # Load and merge data
            looks = pd.read_csv('looks.csv')
            products = pd.read_csv('products.csv')

            # Create product ID to name mapping
            self.product_map = products.set_index('product_id')['product_name'].to_dict()

            # Create look ID to product IDs mapping
            self.look_product_map = looks.groupby('look_id')['product_id'].apply(list).to_dict()

            # Generate product embeddings
            product_embeddings = {}
            for pid, name in self.product_map.items():
                product_embeddings[pid] = self.embedding_model.encode(name)

            # Generate look embeddings
            self.look_embeddings = []
            for pids in self.look_product_map.values():
                embeddings = [product_embeddings[pid] for pid in pids if pid in product_embeddings]
                if embeddings:
                    self.look_embeddings.append(np.mean(embeddings, axis=0))
                else:
                    self.look_embeddings.append(np.zeros(384))  # Handle empty looks

            self.look_embeddings = np.array(self.look_embeddings)

        except Exception as e:
            QMessageBox.critical(self, "Data Error", f"Failed to load data: {str(e)}")
            sys.exit(1)

    def handle_recommendation(self):
        query = self.input_field.text().strip()
        if not query:
            QMessageBox.warning(self, "Input Error", "Please enter a style description")
            return

        self.submit_btn.setEnabled(False)
        self.submit_btn.setText("Processing...")

        # Initialize embedding model in thread-safe way
        if not self.embedding_model:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create and start worker thread
        self.worker = RecommendationWorker(
            self.embedding_model,
            self.look_embeddings,
            self.product_map,
            list(self.look_product_map.values())
        )
        self.worker.query = query
        self.worker.finished.connect(self.show_recommendations)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def show_recommendations(self, products):
        self.submit_btn.setEnabled(True)
        self.submit_btn.setText("Get Recommendations")

        if not products:
            self.results_area.setText("No recommendations found for your query.")
            return

        result_text = "Top Recommendations:\n" + "\n".join(f"• {name}" for name in products)
        self.results_area.setText(result_text)

    def show_error(self, message):
        self.submit_btn.setEnabled(True)
        self.submit_btn.setText("Get Recommendations")
        QMessageBox.critical(self, "Processing Error", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FashionRecommender()
    window.show()
    sys.exit(app.exec())

