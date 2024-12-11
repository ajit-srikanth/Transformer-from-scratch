from __future__ import unicode_literals
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import io
from collections import defaultdict
from functools import partial
import logging
import torch
from tqdm import tqdm
from collections import Counter
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):

    def __init__(self, src_file, tgt_file, vocab_src_file = None, vocab_tgt_file = None, src_lang='en', tgt_lang='fr', max_length=100):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        
        self.src_tokenizer = get_tokenizer('spacy', language=f'{src_lang}_core_web_sm') if src_lang == 'en' else get_tokenizer('spacy', language=f'{src_lang}_core_news_sm')
        self.tgt_tokenizer = get_tokenizer('spacy', language=f'{tgt_lang}_core_news_sm') if tgt_lang == 'fr' else get_tokenizer('spacy', language=f'{src_lang}_core_web_sm')
        
        self.src_sentences = self.load_data(src_file)
        self.tgt_sentences = self.load_data(tgt_file)

        self.vocab_src_sentences = self.load_data(vocab_src_file) if vocab_src_file else self.src_sentences
        self.vocab_tgt_sentences = self.load_data(vocab_tgt_file) if vocab_src_file else self.tgt_sentences
        
        assert len(self.src_sentences) == len(self.tgt_sentences), "Source and target files must have the same number of sentences."
        
        self.src_vocab = self.build_vocab_from_iterator(map(self.src_tokenizer, self.vocab_src_sentences), specials=['<pad>','<unk>', '<sos>', '<eos>'])
        self.tgt_vocab = self.build_vocab_from_iterator(map(self.tgt_tokenizer, self.vocab_tgt_sentences), specials=['<pad>','<unk>', '<sos>', '<eos>'])
                
        self.src_unk_idx = self.src_vocab['<unk>']
        self.tgt_unk_idx = self.tgt_vocab['<unk>']
        
        # print(f"Debug: src_vocab size={len(self.src_vocab)}, tgt_vocab size={len(self.tgt_vocab)}")
        # print(f"Debug: src_unk_idx={self.src_unk_idx}, tgt_unk_idx={self.tgt_unk_idx}")

    def load_data(self, file_path):
        with io.open(file_path, mode='r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def build_vocab_from_iterator(self, iterator, specials=['<pad>','<unk>', '<sos>', '<eos>']):
        counter = Counter()
        with tqdm(unit_scale=0, unit='lines') as t:
            for tokens in iterator:
                counter.update(tokens)
                t.update(1)
        
        # Add special tokens to the counter
        for special in specials:
            counter[special] = 1
        
        word_vocab = Vocab(counter, specials=specials)
        
        return word_vocab

    def __len__(self):
        return len(self.src_sentences)

    def get_index(self, token, vocab, unk_idx):
        return vocab[token]  # This will return unk_idx for unknown tokens

    def __getitem__(self, idx):
        src_tokens = ['<sos>'] + self.src_tokenizer(self.src_sentences[idx])[:self.max_length-2] + ['<eos>']
        tgt_tokens = ['<sos>'] + self.tgt_tokenizer(self.tgt_sentences[idx])[:self.max_length-2] + ['<eos>']
        
        src_indices = [self.get_index(token, self.src_vocab, self.src_unk_idx) for token in src_tokens]
        tgt_indices = [self.get_index(token, self.tgt_vocab, self.tgt_unk_idx) for token in tgt_tokens]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)
    

def getdata(translation_dataset, src_sentences, tgt_sentences, idx):
        src_tokens = ['<sos>'] + translation_dataset.src_tokenizer(src_sentences[idx])[:translation_dataset.max_length-2] + ['<eos>']
        tgt_tokens = ['<sos>'] + translation_dataset.tgt_tokenizer(tgt_sentences[idx])[:translation_dataset.max_length-2] + ['<eos>']
        
        src_indices = [translation_dataset.get_index(token, translation_dataset.src_vocab, translation_dataset.src_unk_idx) for token in src_tokens]
        tgt_indices = [translation_dataset.get_index(token, translation_dataset.tgt_vocab, translation_dataset.tgt_unk_idx) for token in tgt_tokens]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def load_data(self, file_path):
        with io.open(file_path, mode='r', encoding='utf-8') as f:
            return [line.strip() for line in f]

def collate_batch(batch): # aahhh make sure 0 is the pad token lol
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        src_list.append(_src)
        tgt_list.append(_tgt)
    
    src_list = torch.nn.utils.rnn.pad_sequence(src_list, padding_value=0, batch_first=True)
    tgt_list = torch.nn.utils.rnn.pad_sequence(tgt_list, padding_value=0, batch_first=True)
    return src_list, tgt_list

def create_mask(src, tgt):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt_mask.device)
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask


# def translate(model, src_sentence, src_vocab, tgt_vocab, src_tokenizer, device, max_length=50):
#     model.eval()
    
#     # Tokenize the source sentence
#     src_tokens = ['<sos>'] + src_tokenizer(src_sentence)[:max_length-2] + ['<eos>']
    
#     # Convert tokens to indices
#     src_indices = [src_vocab[token] for token in src_tokens]
#     src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
    
#     src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
    
#     encoder_output = model.encoder(src_tensor, src_mask)

#     tgt_tensor = torch.tensor([tgt_vocab['<sos>']]).unsqueeze(0).to(device)
    
#     for _ in tqdm(range(max_length)):
#         tgt_mask = (tgt_tensor != 0).unsqueeze(1).unsqueeze(3)
#         seq_length = tgt_tensor.size(1)

#         tgt_mask = torch.ones(1, 1, seq_length, seq_length).to(device).bool()
        
#         decoder_output = model.decoder(tgt_tensor, encoder_output, src_mask, tgt_mask)
#         output = model.fc(decoder_output[:, -1])
#         _, predicted = torch.max(output, dim=1)
        
#         tgt_tensor = torch.cat([tgt_tensor, predicted.unsqueeze(0)], dim=1)
        
#         if predicted.item() == tgt_vocab['<eos>']:
#             break
    
#     # Convert indices back to tokens using the vocabulary's lookup method
#     translated_tokens = [tgt_vocab.itos[idx.item()] for idx in tgt_tensor[0][1:]]
#     return ' '.join(translated_tokens[:-1])  # Exclude <eos> token


def translate(model, src_sentence, src_vocab, tgt_vocab, src_tokenizer, device, max_length=50):
    model.eval()
    
    # Tokenize the source sentence
    src_tokens = ['<sos>'] + src_tokenizer(src_sentence)[:max_length-2] + ['<eos>']
    
    # Convert tokens to indices
    src_indices = [src_vocab[token] for token in src_tokens]
    src = torch.tensor(src_indices).unsqueeze(0).to(device)

    tgt = torch.tensor([tgt_vocab['<sos>']]).unsqueeze(0).to(device)
    
    for _ in tqdm(range(max_length)):
        # src_mask, tgt_mask = create_mask(src, tgt)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        output = model(src, tgt, src_mask, tgt_mask)
        _, predicted = torch.max(output[:,-1], dim=1)
        
        tgt = torch.cat([tgt, predicted.unsqueeze(0)], dim=1)
        
        if predicted.item() == tgt_vocab['<eos>']:
            break
    
    # Convert indices back to tokens using the vocabulary's lookup method
    translated_tokens = [tgt_vocab.itos[idx.item()] for idx in tgt[0][1:]]
    return ' '.join(translated_tokens[:-1])  # Exclude <eos> token


# def translate(model, src_sentence, src_vocab, tgt_vocab, src_tokenizer, device, max_length=50):
#     model.eval()
    
#     # Tokenize the source sentence
#     src_tokens = ['<sos>'] + src_tokenizer(src_sentence)[:max_length-2] + ['<eos>']
    
#     # Convert tokens to indices
#     src_indices = [src_vocab[token] for token in src_tokens]
#     src = torch.tensor(src_indices).unsqueeze(0).to(device)

#     tgt = torch.tensor([tgt_vocab['<sos>']] + [0] * (max_length - 1)).unsqueeze(0).to(device)
    
#     for i in tqdm(range(max_length-1)):
#         src_mask, tgt_mask = create_mask(src, tgt)
#         output = model(src, tgt, src_mask, tgt_mask)
#         _, predicted = torch.max(output[:,i+1], dim=1)
        
#         tgt[0][i+1] = predicted.item()
        
#         if predicted.item() == tgt_vocab['<eos>']:
#             break
    
#     # Convert indices back to tokens using the vocabulary's lookup method
#     translated_tokens = [tgt_vocab.itos[idx.item()] for idx in tgt[0]]
#     return ' '.join(translated_tokens[1:-1])
