import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from imblearn.over_sampling import SMOTE


# Load embeddings 
entity_embeddings = np.load("entity_embeddings_music.npy")
relation_embeddings = np.load("relation_embeddings_music.npy")

# Normalize the embeddings
scaler = StandardScaler()
entity_embeddings = scaler.fit_transform(entity_embeddings)
relation_embeddings = scaler.fit_transform(relation_embeddings)

# Dataset class
class EmbeddingsDataset(Dataset):
    def __init__(self, entity_embeddings, relation_embeddings, labels):
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entity_emb = self.entity_embeddings[idx]
        relation_emb = self.relation_embeddings[idx]
        label = self.labels[idx]
        return entity_emb, relation_emb, label

# Define Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attn(lstm_output)).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * lstm_output, dim=1)
        return context_vector, attn_weights

# Define LSTM model with attention
class RecommenderLSTM_ANN_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.5):
        super(RecommenderLSTM_ANN_Attention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context_vector, attn_weights = self.attention(lstm_out)
        out = self.fc1(context_vector)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.sigmoid(out)  
        return out

# Combine entity and relation embeddings into pairs
combined_embeddings = [(entity_embeddings[i], relation_embeddings[j]) for i in range(len(entity_embeddings)) for j in range(len(relation_embeddings))]
combined_labels = [np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r)) for e, r in combined_embeddings]

# Convert combined embeddings and labels to arrays
X_combined = np.array(combined_embeddings)
y_combined = np.array([1 if label >= 0.1 else 0 for label in combined_labels])  # Binary labels

# Reshape X_combined to be 2D by flattening entity and relation embeddings
X_combined_flat = X_combined.reshape(X_combined.shape[0], -1)

# Split the dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_combined_flat, y_combined, test_size=0.2, random_state=42)

# Function to apply SMOTE (oversample the minority class in the training set)
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Apply SMOTE only to the training data
X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

# Reshape the resampled dataset back into entities and relations
train_entities = X_train_resampled[:, :entity_embeddings.shape[1]]
train_relations = X_train_resampled[:, entity_embeddings.shape[1]:]

# Prepare the test set entities and relations
test_entities = X_test[:, :entity_embeddings.shape[1]]
test_relations = X_test[:, entity_embeddings.shape[1]:]

# Initialize model parameters
input_dim = entity_embeddings.shape[1]
hidden_dim = 128
num_layers = 3
output_dim = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# K-Fold Cross Validation on the training data
kf = KFold(n_splits=3, shuffle=True)

accuracies, precisions, recalls, f1_scores, auc_rocs = [], [], [], [], []

# K-Fold Cross-Validation Loop on training data
for fold, (train_idx, val_idx) in enumerate(kf.split(train_entities)):
    print(f"Fold {fold + 1}/{kf.n_splits}")

    # Split train and validation data
    X_fold_train_entities, X_fold_val_entities = train_entities[train_idx], train_entities[val_idx]
    X_fold_train_relations, X_fold_val_relations = train_relations[train_idx], train_relations[val_idx]
    y_fold_train, y_fold_val = y_train_resampled[train_idx], y_train_resampled[val_idx]

    # Convert to PyTorch datasets and loaders
    train_dataset = EmbeddingsDataset(X_fold_train_entities, X_fold_train_relations, y_fold_train)
    val_dataset = EmbeddingsDataset(X_fold_val_entities, X_fold_val_relations, y_fold_val)

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = RecommenderLSTM_ANN_Attention(input_dim, hidden_dim, num_layers, output_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()  # Modified loss function to Binary Cross-Entropy Loss

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Training with original data
        for batch in train_dataloader:
            entity_emb, relation_emb, labels = batch
            inputs = torch.stack((entity_emb, relation_emb), dim=1).to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}")

    # Evaluation on validation set
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            entity_emb, relation_emb, labels = batch
            inputs = torch.stack((entity_emb, relation_emb), dim=1).float().to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Convert to binary classification
    threshold = 0.7
    binary_predictions = [1 if pred >= threshold else 0 for pred in predictions]
    binary_true_labels = [1 if label >= threshold else 0 for label in true_labels]

    # Calculate metrics for validation set
    accuracy = accuracy_score(binary_true_labels, binary_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(binary_true_labels, binary_predictions, average='binary')
    auc_roc = roc_auc_score(binary_true_labels, predictions)

    # Append metrics for this fold
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    auc_rocs.append(auc_roc)

# Print average metrics across all folds
print(f"Cross-Validated Accuracy: {np.mean(accuracies):.4f}")
print(f"Cross-Validated Precision: {np.mean(precisions):.4f}")
print(f"Cross-Validated Recall: {np.mean(recalls):.4f}")
print(f"Cross-Validated F1-Score: {np.mean(f1_scores):.4f}")
print(f"Cross-Validated AUC-ROC: {np.mean(auc_rocs):.4f}")

# Evaluation on the test set
test_dataset = EmbeddingsDataset(test_entities, test_relations, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Test set evaluation
model.eval()
test_predictions, test_true_labels = [], []
with torch.no_grad():
    for batch in test_dataloader:
        entity_emb, relation_emb, labels = batch
        inputs = torch.stack((entity_emb, relation_emb), dim=1).float().to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(inputs)
        test_predictions.extend(outputs.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

# Convert to binary classification
test_binary_predictions = [1 if pred >= threshold else 0 for pred in test_predictions]
test_binary_true_labels = [1 if label >= threshold else 0 for label in test_true_labels]

# Calculate metrics for the test set
test_accuracy = accuracy_score(test_binary_true_labels, test_binary_predictions)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_binary_true_labels, test_binary_predictions, average='binary')
test_auc_roc = roc_auc_score(test_binary_true_labels, test_predictions)

# Print the test set metrics
print(f"\nTest Set Accuracy: {test_accuracy:.4f}")
print(f"Test Set Precision: {test_precision:.4f}")
print(f"Test Set Recall: {test_recall:.4f}")
print(f"Test Set F1-Score: {test_f1:.4f}")
print(f"Test Set AUC-ROC: {test_auc_roc:.4f}")