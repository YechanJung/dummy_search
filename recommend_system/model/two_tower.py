import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 16

class QueryTower(nn.Module):
    def __init__(self, user_id_list, gender_list, countries_list):
        super(QueryTower, self).__init__()
        self.emb_dim = DIM
        
        # Create mappings from strings to indices
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_id_list)}
        self.gender_to_index = {gender: idx for idx, gender in enumerate(gender_list)}
        self.country_to_index = {country: idx for idx, country in enumerate(countries_list)}
        
        # User embedding
        self.user_embedding = nn.Embedding(len(user_id_list) + 1, self.emb_dim)
        
        # One-hot encoding dimensions
        self.gender_embedding_dim = len(gender_list)
        self.country_embedding_dim = len(countries_list)
        
        # Normalization parameters (to be set with actual data)
        self.age_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.age_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.sin_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.sin_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.cos_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.cos_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.views_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.views_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.clicks_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.clicks_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        
        # Fully connected neural network
        input_dim = (self.emb_dim + 5 + self.gender_embedding_dim + self.country_embedding_dim)
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
    
    def forward(self, inputs):
        device = next(self.parameters()).device
        
        # Map user_id to index
        user_id_indices = torch.tensor(
            [self.user_id_to_index.get(uid, len(self.user_id_to_index)) for uid in inputs['user_id']],
            dtype=torch.long,
            device=device
        )
        user_embeddings = self.user_embedding(user_id_indices)
        
        # Normalize features
        age = inputs['age'].unsqueeze(1).to(device)
        sin_month = inputs['sin_month'].unsqueeze(1).to(device)
        cos_month = inputs['cos_month'].unsqueeze(1).to(device)
        view_count = inputs['view_count'].unsqueeze(1).to(device)
        click_count = inputs['click_count'].unsqueeze(1).to(device)
        
        # Apply normalization
        age_norm = (age - self.age_mean) / (self.age_std + 1e-6)
        sin_month_norm = (sin_month - self.sin_mean) / (self.sin_std + 1e-6)
        cos_month_norm = (cos_month - self.cos_mean) / (self.cos_std + 1e-6)
        view_count_norm = (view_count - self.views_mean) / (self.views_std + 1e-6)
        click_count_norm = (click_count - self.clicks_mean) / (self.clicks_std + 1e-6)
        
        # One-hot encode gender and country
        gender_indices = torch.tensor(
            [self.gender_to_index.get(gender, 0) for gender in inputs['gender']],
            dtype=torch.long,
            device=device
        )
        gender_one_hot = F.one_hot(gender_indices, num_classes=self.gender_embedding_dim).float()
        
        country_indices = torch.tensor(
            [self.country_to_index.get(country, 0) for country in inputs['country']],
            dtype=torch.long,
            device=device
        )
        country_one_hot = F.one_hot(country_indices, num_classes=self.country_embedding_dim).float()
        
        # Concatenate all inputs
        concatenated_inputs = torch.cat([
            user_embeddings,
            age_norm,
            sin_month_norm,
            cos_month_norm,
            view_count_norm,
            click_count_norm,
            gender_one_hot,
            country_one_hot
        ], dim=1)
        
        outputs = self.fnn(concatenated_inputs)
        return outputs

class ItemTower(nn.Module):
    def __init__(self, item_id_list, category_list):
        super(ItemTower, self).__init__()
        self.emb_dim = DIM
        
        # Create mappings from strings to indices
        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_id_list)}
        self.category_to_index = {category: idx for idx, category in enumerate(category_list)}
        
        # Item embedding
        self.item_embedding = nn.Embedding(len(item_id_list) + 1, self.emb_dim)
        
        # One-hot encoding dimensions
        self.category_embedding_dim = len(category_list)
        
        # Normalization parameters
        self.views_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.views_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.clicks_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.clicks_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.title_length_mean = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.title_length_std = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        
        # Fully connected neural network
        input_dim = self.emb_dim + self.category_embedding_dim + 3
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
    
    def forward(self, inputs):
        device = next(self.parameters()).device
        
        # Map item_id to index
        item_id_indices = torch.tensor(
            [self.item_id_to_index.get(iid, len(self.item_id_to_index)) for iid in inputs['item_id']],
            dtype=torch.long,
            device=device
        )
        item_embeddings = self.item_embedding(item_id_indices)
        
        # Normalize features
        view_count = inputs['view_count'].unsqueeze(1).to(device)
        click_count = inputs['click_count'].unsqueeze(1).to(device)
        title_length = inputs['title_length'].unsqueeze(1).to(device)
        
        # Apply normalization
        view_count_norm = (view_count - self.views_mean) / (self.views_std + 1e-6)
        click_count_norm = (click_count - self.clicks_mean) / (self.clicks_std + 1e-6)
        title_length_norm = (title_length - self.title_length_mean) / (self.title_length_std + 1e-6)
        
        # One-hot encode category
        category_indices = torch.tensor(
            [self.category_to_index.get(cat, 0) for cat in inputs['category']],
            dtype=torch.long,
            device=device
        )
        category_one_hot = F.one_hot(category_indices, num_classes=self.category_embedding_dim).float()
        
        # Concatenate all inputs
        concatenated_inputs = torch.cat([
            item_embeddings,
            category_one_hot,
            view_count_norm,
            click_count_norm,
            title_length_norm
        ], dim=1)
        
        outputs = self.fnn(concatenated_inputs)
        return outputs

class TwoTowerModel(nn.Module):
    def __init__(self, query_model, item_model):
        super(TwoTowerModel, self).__init__()
        self.query_model = query_model
        self.item_model = item_model
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, batch):
        user_embeddings = self.query_model(batch)
        item_embeddings = self.item_model(batch)
        return user_embeddings, item_embeddings
    
    def compute_loss(self, user_embeddings, item_embeddings):
        # Compute logits
        logits = user_embeddings @ item_embeddings.T  # Shape: (batch_size, batch_size)
        labels = torch.arange(user_embeddings.size(0)).to(user_embeddings.device)
        loss = self.loss_fn(logits, labels)
        return loss

def train_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad()
    user_embeddings, item_embeddings = model(batch)
    loss = model.compute_loss(user_embeddings, item_embeddings)
    loss.backward()
    optimizer.step()
    return {'loss': loss.item()}

def test_step(model, batch):
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(batch)
        loss = model.compute_loss(user_embeddings, item_embeddings)
    return {'loss': loss.item()}
