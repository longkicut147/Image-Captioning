from requirements import *
from encode import *
from model import TransformerDecoder


# test dataset
images_list, captions_list = get_img_and_caption(test_images_path, captions_path)
test_features = CocoDataset(images_list=images_list, captions_list=captions_list)

# test dataloader
test_dataloader = get_loader(test_features, split='test')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
decoder = TransformerDecoder(vocab_size=vocab_size).to(device)
decoder.load_state_dict(torch.load(weights_path, map_location=device))
decoder.eval()

max_seq_len = 20

for i, (img_features, _) in enumerate(test_dataloader):
    # Lấy caption gốc
    Image._show(Image.open(test_features.images_list[i]).convert("RGB"))

    # Caption mới bắt đầu với token CLS
    caption_generating = [tokenizer.cls_token_id]

    img_features = img_features.to(device)
    with torch.no_grad():
        for _ in range(max_seq_len):
            caption_tensor = torch.tensor([caption_generating], device=device, dtype=torch.long)

            # Chạy forward: dựa vào ảnh và token đã sinh ra tới thời điểm hiện tại để dự đoán token tiếp theo
            logits = decoder(img_features, caption_tensor)

            # Tại mỗi bước, lấy token có xác suất cao nhất và thêm vào chuỗi caption hoàn chỉnh
            next_token = logits[0, -1].argmax().item()
            caption_generating.append(next_token)

            # Nếu gặp token kết thúc [SEP], dừng lại
            if next_token == tokenizer.sep_token_id:
                break

    # Giải mã từ danh sách token thành câu
    caption_generated = tokenizer.decode(caption_generating, skip_special_tokens=True)

    print(f"Generated caption: {caption_generated}\n")

    

