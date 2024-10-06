import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Charger les triplets
triplets = pd.read_csv('./data/graph_triplets_from_kkbox.csv')
triplets.columns = ['source', 'relation', 'target']

# Créer des mappings
entity_to_id = {}
relation_to_id = {}
current_entity_id = 0
current_relation_id = 0

for index, row in triplets.iterrows():
    head, relation, tail = row['source'], row['relation'], row['target']
    if head not in entity_to_id:
        entity_to_id[head] = current_entity_id
        current_entity_id += 1
    if tail not in entity_to_id:
        entity_to_id[tail] = current_entity_id
        current_entity_id += 1
    if relation not in relation_to_id:
        relation_to_id[relation] = current_relation_id
        current_relation_id += 1

num_entities = len(entity_to_id)
num_relations = len(relation_to_id)

class TransR(nn.Module):
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim):
        super(TransR, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)
        self.projection_matrices = nn.Parameter(torch.Tensor(num_relations, entity_dim, relation_dim))

        # Initialiser les embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        nn.init.xavier_uniform_(self.projection_matrices)

    def forward(self, h, r, t):
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)

        proj_h_emb = torch.bmm(h_emb.unsqueeze(1), self.projection_matrices[r]).squeeze(1)
        proj_t_emb = torch.bmm(t_emb.unsqueeze(1), self.projection_matrices[r]).squeeze(1)

        return proj_h_emb, r_emb, proj_t_emb

    def score_function(self, h_emb, r_emb, t_emb):
        return torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)

# Hyperparamètres
entity_dim = 100
relation_dim = 100
margin = 2.0
learning_rate = 0.001  # Réduire le taux d'apprentissage
num_epochs = 50
batch_size = 256
num_workers = 2

# Définir le modèle TransR
transR = TransR(num_entities, num_relations, entity_dim, relation_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transR = transR.to(device)
optimizer = optim.Adam(transR.parameters(), lr=learning_rate, weight_decay=1e-4)  # Augmenter la régularisation

def generate_negative_samples(triplets, num_entities):
    negative_triplets = []
    for head, relation, tail in triplets:
        while True:
            if np.random.rand() < 0.5:
                head = np.random.randint(0, num_entities)
            else:
                tail = np.random.randint(0, num_entities)
            if (head, relation, tail) not in triplets:
                break
        negative_triplets.append((head, relation, tail))
    return negative_triplets

triplet_indices = [(entity_to_id[h], relation_to_id[r], entity_to_id[t]) for h, r, t in triplets.values]

for epoch in range(num_epochs):
    np.random.shuffle(triplet_indices)
    epoch_loss = 0

    with tqdm(total=len(triplet_indices), desc=f"Training Epoch {epoch+1}", unit="triplet") as pbar:
        for i in range(0, len(triplet_indices), batch_size):
            batch = triplet_indices[i:i+batch_size]
            heads, relations, tails = zip(*batch)
            heads = torch.LongTensor(heads).to(device)
            relations = torch.LongTensor(relations).to(device)
            tails = torch.LongTensor(tails).to(device)

            negative_batch = generate_negative_samples(batch, num_entities)
            n_heads, n_relations, n_tails = zip(*negative_batch)
            n_heads = torch.LongTensor(n_heads).to(device)
            n_relations = torch.LongTensor(n_relations).to(device)
            n_tails = torch.LongTensor(n_tails).to(device)

            optimizer.zero_grad()

            # Échantillons positifs
            pos_h_emb, pos_r_emb, pos_t_emb = transR(heads, relations, tails)
            pos_scores = transR.score_function(pos_h_emb, pos_r_emb, pos_t_emb)

            # Échantillons négatifs
            neg_h_emb, neg_r_emb, neg_t_emb = transR(n_heads, n_relations, n_tails)
            neg_scores = transR.score_function(neg_h_emb, neg_r_emb, neg_t_emb)

            # Perte basée sur la marge
            loss = torch.mean(torch.relu(pos_scores - neg_scores + margin))
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            pbar.update(len(batch))

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(triplet_indices)}")

# Obtenir les embeddings
entity_embeddings = transR.entity_embeddings.weight.detach().cpu().numpy()
relation_embeddings = transR.relation_embeddings.weight.detach().cpu().numpy()

# Mapper les ID aux noms des entités et des relations si nécessaire
id_to_entity = {v: k for k, v in entity_to_id.items()}
id_to_relation = {v: k for k, v in relation_to_id.items()}

np.save("entity_embeddings_music.npy", entity_embeddings)
np.save("relation_embeddings_music.npy", relation_embeddings)

print("Les embeddings des entités et des relations ont été sauvegardés.")