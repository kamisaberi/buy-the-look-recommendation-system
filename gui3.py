import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLineEdit, QTextEdit, QPushButton, QLabel
from PyQt6.QtCore import Qt

# --- Step 1: Load and Merge Data ---
looks_df = pd.read_csv("looks.csv")  # Columns: look_id, category, product_id
products_df = pd.read_csv("products.csv")  # Columns: product_id, product_name
merged_df = looks_df.merge(products_df, on="product_id")

# Group by look_id to aggregate product names and get category
looks = merged_df.groupby("look_id").agg({
    "category": "first",
    "product_name": lambda x: ", ".join(x)
}).reset_index()

# --- Step 2: Generate Synthetic Descriptions ---
templates = [
    "I want a {} look with {}.",
    "Looking for a {} outfit: {}.",
    "Can you recommend a {} style including {}?",
    "I need a {} look: {}.",
    "Show me a {} outfit with {}."
]

descriptions = []
for _, row in looks.iterrows():
    category = row["category"]
    products = row["product_name"]
    for template in templates:
        desc = template.format(category, products)
        descriptions.append({"look_id": row["look_id"], "description": desc})

descriptions_df = pd.DataFrame(descriptions)

# --- Step 3: Split into Training and Validation Sets ---
train_df, val_df = train_test_split(descriptions_df, test_size=0.2, random_state=42)

# --- Step 4: Build Vocabulary ---
def tokenize(text):
    return text.lower().split()

word_counts = Counter()
for desc in train_df["description"]:
    tokens = tokenize(desc)
    word_counts.update(tokens)

min_freq = 2
vocab = [word for word, count in word_counts.items() if count >= min_freq]
vocab = ["<PAD>", "<UNK>"] + vocab
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# --- Step 5: Convert Text to Sequences ---
def text_to_sequence(text, word_to_idx):
    tokens = tokenize(text)
    return [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in tokens]

max_len = max(len(text_to_sequence(desc, word_to_idx)) for desc in descriptions_df["description"])

def pad_sequences(sequences, max_len):
    padded = [seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]
    return torch.tensor(padded)

train_sequences = [text_to_sequence(desc, word_to_idx) for desc in train_df["description"]]
val_sequences = [text_to_sequence(desc, word_to_idx) for desc in val_df["description"]]
X_train = pad_sequences(train_sequences, max_len)
X_val = pad_sequences(val_sequences, max_len)

# --- Step 6: Prepare Labels ---
unique_looks = looks["look_id"].unique()
look_to_idx = {look: idx for idx, look in enumerate(unique_looks)}
y_train = torch.tensor([look_to_idx[look] for look in train_df["look_id"]])
y_val = torch.tensor([look_to_idx[look] for look in val_df["look_id"]])

# --- Step 7: Define the Model ---
class LookClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LookClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]
        output = self.fc(hidden)
        return output

# Initialize model
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
num_classes = len(unique_looks)
model = LookClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)

# --- Step 8: Train the Model ---
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()

    val_loss /= len(val_loader)
    accuracy = correct / len(val_dataset)
    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

# --- Step 9: Inference Functions ---
def predict_look(user_request, model, word_to_idx, max_len, look_to_idx):
    seq = text_to_sequence(user_request, word_to_idx)
    padded = torch.tensor(seq[:max_len] + [0] * (max_len - len(seq)))
    padded = padded.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(padded)
        pred = torch.argmax(output, dim=1).item()
    idx_to_look = {idx: look for look, idx in look_to_idx.items()}
    return idx_to_look[pred]

def recommend_look(user_request, model, word_to_idx, max_len, look_to_idx, looks_df):
    predicted_look_id = predict_look(user_request, model, word_to_idx, max_len, look_to_idx)
    look = looks_df[looks_df["look_id"] == predicted_look_id].iloc[0]
    category = look["category"]
    products = look["product_name"]
    description = f"{category} look: {products}"
    return description, products.split(", ")

# --- Step 10: PyQt6 Application ---
class FashionRecommender(QMainWindow):
    def __init__(self, model, word_to_idx, max_len, look_to_idx, looks_df):
        super().__init__()
        self.model = model
        self.word_to_idx = word_to_idx
        self.max_len = max_len
        self.look_to_idx = look_to_idx
        self.looks_df = looks_df
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Fashion Look Recommender")
        self.setGeometry(100, 100, 600, 400)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Prompt label and input
        self.prompt_label = QLabel("Enter your fashion request (e.g., 'I want a casual outfit with a blue top'):")
        layout.addWidget(self.prompt_label)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Type your request here")
        layout.addWidget(self.prompt_input)

        # Recommend button
        self.recommend_button = QPushButton("Recommend")
        self.recommend_button.clicked.connect(self.show_recommendation)
        layout.addWidget(self.recommend_button)

        # Output display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setPlaceholderText("Recommendation will appear here")
        layout.addWidget(self.output_display)

    def show_recommendation(self):
        user_request = self.prompt_input.text().strip()
        if not user_request:
            self.output_display.setText("Please enter a request.")
            return

        try:
            description, products = recommend_look(
                user_request, self.model, self.word_to_idx, self.max_len, self.look_to_idx, self.looks_df
            )
            output = f"Recommended: {description}\n\nClothing Items:\n"
            for item in products:
                output += f"- {item.strip()}\n"
            self.output_display.setText(output)
        except Exception as e:
            self.output_display.setText(f"Error: {str(e)}")

# --- Step 11: Run the Application ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FashionRecommender(model, word_to_idx, max_len, look_to_idx, looks)
    window.show()
    sys.exit(app.exec())