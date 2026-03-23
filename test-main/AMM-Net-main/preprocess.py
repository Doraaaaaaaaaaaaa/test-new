import torch
from torchvision import transforms
from transformers import BertTokenizer
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_LEN = 200
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return _tokenizer


def txt_process(txt):
    tokenizer = get_tokenizer()

    def pad(x):
        if len(x) > MAX_LEN:
            x = x[:MAX_LEN]
        else:
            x = x + [0] * (MAX_LEN - len(x))
        return x

    sentences = '[CLS] ' + txt + ' [SEP]'
    tokenized_sents = tokenizer.tokenize(sentences)
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_sents)
    input_ids = pad(input_ids)
    input_ids = torch.tensor(input_ids)
    return input_ids


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.Resize(size=(448, 448)),
    transforms.ToTensor(),
    normalize
])

clip_normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711]
)

transform_att = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    clip_normalize
])
