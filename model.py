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

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads,dropout=dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)

        # Projection from image features (2048) to hidden_dim (768)
        self.img_projection = nn.Linear(2048, hidden_dim)

        # Final classification layer to predict vocab
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img_features, input_tokens):
        """
        img_features: [B, 2048]
        input_tokens: [B, seq_len]
        """
        B, seq_len = input_tokens.shape
        # Tạo token embedding
        token_embedding = self.token_embedding(input_tokens)
        # Tạo position embedding
        positions = torch.arange(0, seq_len, device=input_tokens.device).unsqueeze(0).expand(B, seq_len)
        position_embedding = self.position_embedding(positions)
        # Tổng hợp embedding (token + position)
        decoder_input_embedding = token_embedding + position_embedding  # [B, seq_len, hidden_dim]


        # Điều chỉnh img_features thành [B, 1, hidden_dim] để làm memory cho transformer decoder
        img_memory = self.img_projection(img_features).unsqueeze(1)


        # Transformer decoder forward
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_tokens.device)
        output = self.transformer_decoder(decoder_input_embedding, memory=img_memory, tgt_mask=tgt_mask)  # [B, seq_len, hidden_dim]

        # Dự đoán vocab
        logits = self.fc_out(output)  # [B, seq_len, vocab_size]
        return logits
    

def inference(image, decoder, tokenizer, device, max_seq_len=20):
    """
    Sinh caption cho một ảnh.
    Args: image: image_features
    """
    for i, (img_features, _) in enumerate(image):
        Image._show(Image.open(image.images_list[i]).convert("RGB"))

        # Caption mới bắt đầu với token CLS
        caption_generating = [tokenizer.cls_token_id]
        caption_generating = caption_generating.sq

        # thêm một chiều batch_size
        img_features = img_features.squeeze(0)
        img_features = img_features.to(device)

        with torch.no_grad():
            for _ in range(max_seq_len):
                caption_tensor = torch.tensor([caption_generating], device=device, dtype=torch.long)

                # Chạy forward: dựa vào ảnh và token đã sinh ra tới thời điểm hiện tại để dự đoán token tiếp theo
                logits = decoder(img_features, caption_tensor)

                # Tại mỗi bước, lấy token có xác suất cao nhất và thêm vào caption hoàn chỉnh
                next_token = logits[0, -1].argmax().item()
                caption_generating.append(next_token)

                # Nếu gặp token kết thúc [SEP], dừng lại
                if next_token == tokenizer.sep_token_id:
                    break

        # Giải mã từ danh sách token thành câu
        caption_generated = tokenizer.decode(caption_generating, skip_special_tokens=True)

        print(f"Generated caption: {caption_generated}\n")