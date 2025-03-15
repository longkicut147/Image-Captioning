from requirements import *
from model import ResNet50Encoder

'''
This file contains functions to encode images and captions to be used in training
'''

def get_img_and_caption(images_path, captions_path):
        '''
        args: images_path, captions_path
        return: list images, captions sorted by id
        '''
        # convert json file to dataframe
        with open(captions_path, 'r') as f:
            captions_data = json.load(f)
        df = pd.DataFrame(captions_data)

        images_list = []
        captions_list = []

        for i in os.listdir(images_path):
            img_path = images_path + '/' + i
            images_list.append(img_path)
            # get image id from image file name
            file_name = os.path.basename(img_path)
            image_id = file_name.split('_')[-1].split('.')[0]
            # get a list of 5 captions relative to image id
            caption = df[df['image_id'] == int(image_id)]['caption'].values
            captions_list.append(list(caption))

        return images_list, captions_list



class CocoDataset(Dataset):
    def __init__(self, images_list, captions_list):
        """
        images_list: A list containing images link
        caption_list: A list containing captions corresponding to the images

        getitem: train: tensor contains image features encoded and tensor contains 1 random tokenized captions
                test: tensor contains image features encoded and tensor contains 5 tokenized captions
        """
        self.images_list = images_list
        self.captions_list = captions_list
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        
        self.captions = []  # tokenized captions
        self.caplen = []    # tokenized captions length
        self.img_features = []  # image features

        self.encoder = ResNet50Encoder().eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # extract and save image features
        for img_path in images_list:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)   # [1, 3, 224, 224]
            with torch.no_grad():
                feature = self.encoder(image_tensor).squeeze(0) # [3, 224, 224] -> [2048]
            self.img_features.append(feature)

        # 5 captions for each image
        if captions_list is not None:
            for list_cap in captions_list:
                tokenized_list = []
                length_list = []
                for caption in list_cap:
                    tokenized_caption = self.tokenizer.encode(caption, add_special_tokens=True)
                    tokenized_list.append(tokenized_caption)
                    length_list.append(len(tokenized_caption))
                self.captions.append(tokenized_list)
                self.caplen.append(length_list)
        

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        feature = self.img_features[idx]
        # 1 random caption for training
        caption = torch.tensor(self.captions[idx][random.randrange(len(self.captions[idx]))], dtype=torch.long)
        return feature, caption



class CaptionLengthSampler(Sampler):
    """
    Sampler tạo batch dựa trên độ dài caption.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        # Sort indices by caption length
        self.indices.sort(key=lambda i: dataset.caplen[i])

    def __iter__(self):
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        batches = [batch for batch in batches if len(batch) == self.batch_size]
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.indices) // self.batch_size

def collate_fn(batch):
    """
    Collate function để gộp batch, pad caption để cùng chiều dài.
    """
    images, captions = zip(*batch)

    images = torch.stack(images, dim=0)  # (batch_size, C, H, W)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)  # (batch_size, max_len)

    return images, captions



def get_loader(dataset, split='train', batch_size=32):
    if split == 'test':
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader
    sampler = CaptionLengthSampler(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    return dataloader