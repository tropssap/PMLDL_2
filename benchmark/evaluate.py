import pandas as pd
import torch
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch.nn import Dropout
from torch_geometric.nn import GATv2Conv, Linear, to_hetero
from torch_geometric.utils import dropout_edge
from sentence_transformers import SentenceTransformer


metadata = ['user', 'item'], [('user', 'rates', 'item'), ('item', 'rev_rates', 'user')]
def preprocess_data():
    # Read and save items and users data
    items = pd.read_csv("data/interim/items.csv")
    items.to_csv("benchmark/data/items.csv", index=False)
    users = pd.read_csv("data/interim/users.csv")
    users.to_csv("benchmark/data/users.csv", index=False)

    # Read, process, and save ratings data
    ratings_base = pd.read_csv("data/ml-100k/u1.base", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    del ratings_base['timestamp']
    ratings_base.to_csv("benchmark/data/ratings-base.csv", index=False)
    ratings_base = pd.read_csv("data/ml-100k/u1.test", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    del ratings_base['timestamp']
    ratings_base.to_csv("benchmark/data/ratings-test.csv", index=False)

# preprocess_data()

def load_dataset(device, users, items, ratings, genres):
    # Create torch edges from ratings
    def create_torch_edges(ratings):
        src = ratings["user_id"] - 1
        dst = ratings["item_id"] - 1
        attrs = ratings["rating"]

        edge_index = torch.tensor([src, dst], dtype=torch.int64)
        edge_attr = torch.tensor(attrs)

        return edge_index, edge_attr

    edge_index, edge_attr = create_torch_edges(ratings)

    # Encode movie titles using SentenceTransformer
    def SequenceEncoder(movie_titles, model_name=None):
        model = SentenceTransformer(model_name, device=device)
        title_embeddings = model.encode(movie_titles, show_progress_bar=True,
                                        convert_to_tensor=True, device=device)

        return title_embeddings.to("cpu")

    item_title = SequenceEncoder(items["movie_title"], model_name='all-MiniLM-L6-v2')
    item_genres = torch.tensor(items[genres.name].to_numpy(), dtype=torch.bool)

    item_x = torch.cat((item_title, item_genres), dim=-1).float()

    # Process user data
    user_ages = torch.tensor(users["age"].to_numpy()[:, np.newaxis], dtype=torch.uint8)
    user_sex = torch.tensor(users[["male", "female"]].to_numpy(), dtype=torch.bool)
    occupations = [i for i in users.keys() if i.startswith("occupation_")]
    user_occupation = torch.tensor(users[occupations].to_numpy(), dtype=torch.bool)
    user_x = torch.cat((user_ages, user_sex, user_occupation), dim=-1).float()

    # Create HeteroData object and process data
    data = HeteroData()

    data['user'].x = user_x
    data['item'].x = item_x
    data['user', 'rates', 'item'].edge_index = edge_index
    data['user', 'rates', 'item'].edge_label = edge_attr

    data = ToUndirected()(data)
    del data['item', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.
    data = data.to(device)

    return data
   


class GNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), 32, add_self_loops=False)
        self.conv2 = GATv2Conv((-1, -1), 32, add_self_loops=False)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)
        
    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['item'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder()
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')  # Convert encoder to hetero
        self.decoder = EdgeDecoder(hidden_channels)
        
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        edge_label_index, mask = dropout_edge(edge_label_index, p=0.25, training=self.training)
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index), mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("models/model.pt")
model.eval()

users = pd.read_csv("benchmark/data/users.csv")
items = pd.read_csv("benchmark/data/items.csv")
ratings = pd.read_csv("benchmark/data/ratings-base.csv")
test_ratings = pd.read_csv("benchmark/data/ratings-test.csv")
genres = pd.read_csv("benchmark/data/u.genre", delimiter="|", names=["name","index"])

data = load_dataset(device, users, items, ratings, genres)

test_ratings = test_ratings.to_numpy()
users = test_ratings[:, 0] - 1
movies = test_ratings[:, 1] - 1
true_labels = torch.tensor(test_ratings[:, 2]).to(device)

x = torch.from_numpy(np.stack([users, movies], axis=0)).to(device)
pred_labels, _ = model(data.x_dict, data.edge_index_dict, x)

print(f"RMSE: {(true_labels - pred_labels).pow(2).mean().sqrt()}")
print(f"MAE: {torch.abs(true_labels - pred_labels).mean()}")
