from requirements import *
from encode import *
from model import LSTMDecoder


# Load features set if exists, otherwise create new features set
if not os.path.exists(features_path):
    images_list, captions_list = get_img_and_caption(train_images_path, captions_path)
    features = CocoDataset(images_list=images_list, captions_list=captions_list)
    dataloader = get_loader(features, split='train')
    with open(features_path, 'wb') as f:
            pickle.dump(features, f)
else:
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
        dataloader = get_loader(features, split='train')


# ----------------- Training ----------------- #


# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = LSTMDecoder(vocab_size=vocab_size).to(device)

# Optimizer, Loss
optimizer = optim.Adam(decoder.parameters(), lr=1e-4)  # Chỉ optimize decoder
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training Loop
num_epochs = 50

for epoch in range(num_epochs):
    decoder.train()

    epoch_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for img_features, captions in pbar:
        img_features = img_features.to(device)  # [B, 2048]
        captions = captions.to(device)          # [B, seq_len]

        # Đầu vào decoder là toàn bộ caption (trừ token cuối)
        decoder_input = captions[:, :-1]  # [B, seq_len-1]
        # Mục tiêu (label) là toàn bộ caption (trừ token đầu tiên)
        labels = captions[:, 1:]  # [B, seq_len-1]

        # Decode caption từ feature ảnh
        logits = decoder(img_features, decoder_input)

        # Tính loss
        loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        epoch_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(dataloader):.4f}")


# Save model
torch.save(decoder.state_dict(), lstm_weights_path)
print(f"Model saved at {lstm_weights_path}")

# Load chỉ trọng số để predict
# decoder.load_state_dict(torch.load(weights_path))
# decoder.eval()