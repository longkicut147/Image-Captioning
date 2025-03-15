from requirements import *

'''
This file contains the CNN encoder and Transformer decoder architecture
'''

class ResNet50Encoder(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1'):
        super(ResNet50Encoder, self).__init__()
        resnet = models.resnet50(weights=weights)

        modules = list(resnet.children())[:-1]    # Loại bỏ lớp FC
        self.encoder = nn.Sequential(*modules)    # Gộp các layer thành một module encoder


    def forward(self, images):
        features = self.encoder(images)
        features = features.flatten(start_dim=1)  # [B,2048]
        return features


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, num_layers=6, num_heads=8, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)  # max length 512

        # Projection from image features (2048) to hidden_dim (768)
        self.img_projection = nn.Linear(2048, hidden_dim)

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads,dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)

        # Final classification layer to predict vocab
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img_features, input_tokens):
        """
        img_features: [B, 2048]
        input_tokens: [B, seq_len]
        """
        B, seq_len = input_tokens.shape
        # Tạo token embedding
        embedded_token = self.token_embedding(input_tokens)
        
        # Tạo ra tensor chứa chỉ số vị trí từ 0 đến seq_len
        positions = torch.arange(seq_len, device=input_tokens.device)   # [0, 1, 2, 3, ..., seq_len] has shape [seq_len]
        # Mở rộng vị trí thành [B, seq_len]
        positions = positions.unsqueeze(0).expand(B, -1)                # [B, seq_len]  
        # embedding position  
        embedded_position = self.position_embedding(positions)          # [B, seq_len, hidden_dim]
        

        # Tổng hợp embedding (token + position) để làm target của transformer decoder
        token_target = embedded_token + embedded_position               # [B, seq_len, hidden_dim]
        # Embedding position cho img_features để làm memory của transformer decoder với seq_len = 1
        img_memory = self.img_projection(img_features).unsqueeze(1)     # [B, seq_len, hidden_dim]


        # Transformer decoder forward
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_tokens.device)
        output = self.transformer_decoder(token_target, memory=img_memory, tgt_mask=tgt_mask)       # [B, seq_len, hidden_dim]

        # Dự đoán vocab
        logits = self.fc_out(output)  # [B, seq_len, vocab_size]
        return logits
    

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, num_layers=2, dropout=0.1):
        super(LSTMDecoder, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)  # max length 512
        
        # Projection from image features (2048) to hidden_dim (768)
        self.img_projection = nn.Linear(2048, hidden_dim)
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Final classification layer to predict vocab
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, img_features, input_tokens):
        """
        img_features: [B, 2048]
        input_tokens: [B, seq_len]
        """
        B, seq_len = input_tokens.shape
        
        # Token embedding
        embedded_token = self.token_embedding(input_tokens)
        
        # Position embedding
        positions = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0).expand(B, -1)
        embedded_position = self.position_embedding(positions)
        
        # Summing token and position embeddings
        token_input = embedded_token + embedded_position  # [B, seq_len, hidden_dim]
        
        # Project image features and use as initial hidden state
        img_memory = self.img_projection(img_features).unsqueeze(0)  # [1, B, hidden_dim]
        h0 = img_memory.expand(self.lstm.num_layers, -1, -1).contiguous()
        c0 = torch.zeros_like(h0)  # Initial cell state
        
        # LSTM forward pass
        lstm_output, _ = self.lstm(token_input, (h0, c0))  # [B, seq_len, hidden_dim]
        
        # Predict vocab
        logits = self.fc_out(lstm_output)  # [B, seq_len, vocab_size]
        return logits
