import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Load the datasets
transactions = pd.read_csv('bank_transactions.csv')
customers = pd.read_csv('Customer DataSet.csv')
stock_data = pd.read_csv('National_Stock_Exchange_of_India_Ltd.csv')

# Merge datasets based on CustomerID
merged_data = pd.merge(transactions, customers, left_on='CustomerID', right_on='CUST_ID', how='inner')

# Check if 'Symbol' column exists in stock_data
if 'Symbol' in stock_data.columns:
    # Check if 'Symbol' column exists in merged_data
    if 'Symbol' in merged_data.columns:
        # Merge with stock data based on 'Symbol'
        merged_data = pd.merge(merged_data, stock_data, on='Symbol', how='inner')
    else:
        print("'Symbol' column not found in merged_data. Skipping merge with stock data.")
else:
    print("'Symbol' column not found in stock_data. Skipping merge with stock data.")

# Feature engineering
scaler = StandardScaler()
label_encoder = LabelEncoder()
numerical_columns = ['CustAccountBalance', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
                     'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                     'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                     'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

# Check if the stock data columns exist in the merged_data
stock_data_columns = ['Open', 'High', 'Low', 'LTP', 'Chng', '% Chng', 'Volume (lacs)', 'Turnover (crs.)', '52w H', '52w L', '365 d % chng']
for col in stock_data_columns:
    if col in merged_data.columns:
        numerical_columns.append(col)

if numerical_columns:
    merged_data_numerical = merged_data[numerical_columns]
    if merged_data_numerical.shape[0] > 0:
        merged_data_numerical = scaler.fit_transform(merged_data_numerical)
        merged_data[numerical_columns] = merged_data_numerical

# Define feature columns and target columns
feature_columns = numerical_columns
target_columns = ['TransactionAmount (INR)']

# Split the data into training and validation sets
if len(merged_data) > 1:
    X_train, X_val, y_train, y_val = train_test_split(merged_data[feature_columns], merged_data[target_columns], test_size=0.2, random_state=42)
else:
    # If the data size is too small, use the entire dataset for training
    X_train, y_train = merged_data[feature_columns], merged_data[target_columns]
    X_val, y_val = None, None

# Convert feature columns to integer indices
if len(X_train) > 0:
    X_train_indices = torch.tensor(label_encoder.fit_transform(X_train.astype(str)), dtype=torch.long)
else:
    X_train_indices = None

if X_val is not None and len(X_val) > 0:
    X_val_indices = torch.tensor(label_encoder.transform(X_val.astype(str)), dtype=torch.long)
else:
    X_val_indices = None

# Convert target columns to PyTorch tensors
if len(y_train) > 0:
    y_train_tensor = torch.Tensor(y_train.values)
else:
    y_train_tensor = None

if y_val is not None and len(y_val) > 0:
    y_val_tensor = torch.Tensor(y_val.values)
else:
    y_val_tensor = None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.einsum('nqhd,nkhd->nhqk', [Q, K])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        attention = torch.nn.functional.softmax(energy / (self.d_model ** (1 / 2)), dim=3)
        x = torch.einsum('nhql,nlhd->nqhd', [attention, V]).permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        x = self.fc_out(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.attention(x, x, x, mask)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)
        feedforward_output = self.feedforward(x)
        x = x + self.dropout(feedforward_output)
        x = self.norm2(x)
        return x

class FinancialTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, num_heads=4, ff_hidden_dim=256, dropout=0.1):
        super(FinancialTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_size, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = x + self.positional_encoding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        x = self.fc(x[-1, :, :])
        return x

model = FinancialTransformer(input_size=len(feature_columns), hidden_size=256, output_size=len(target_columns))

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10

if X_train_indices is not None and y_train_tensor is not None:
    for epoch in range(epochs):
        outputs = model(X_train_indices)

        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# Evaluation on the validation set
if X_val_indices is not None and y_val_tensor is not None:
    with torch.no_grad():
        val_outputs = model(X_val_indices)
        val_loss = criterion(val_outputs, y_val_tensor)
        print(f'Validation Loss: {val_loss.item()}')

# Save the model for deployment
torch.save(model.state_dict(), 'financial_transformer_model.pth')
