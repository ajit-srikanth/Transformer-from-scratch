import torch
from torch.utils.data import DataLoader
from encoder import Encoder
from decoder import Decoder
from utils import TranslationDataset, collate_batch, create_mask, translate
from train import Transformer
import argparse
from tqdm import tqdm
import pdb
import os
import io
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')


def test(model, test_loader, criterion, src_vocab, tgt_vocab, device):
    if os.path.exists("test_translated_full.txt"):
        os.remove("test_translated_full.txt")
    if os.path.exists("translated_test.txt"):
        os.remove("translated_test.txt")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = create_mask(src, tgt)
            
            output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :, :])
            _, predicted = torch.max(output, dim=2)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
            src_sentences = []
            translated_sentences = []
            for s in range(predicted.size(0)):
                src_sentence = []
                for idx in src[s]:
                    word = src_vocab.itos[idx.item()]
                    src_sentence.append(word)
                    if word == '<eos>':
                        break
                # print(src_sentence)
                src_sentences.append(src_sentence)
                # translated_sentences = [tgt_vocab.itos[idx.item()] for idx in predicted[s][1:]] # first sentence in batch
                translated_sentence = []
                for idx in predicted[s]:
                    word = tgt_vocab.itos[idx.item()]
                    translated_sentence.append(word)
                    if word == '<eos>':
                        break
                translated_sentences.append(translated_sentence)
                # print(translated_sentences)
            
            with open("test_translated_full.txt", "a") as f:
                for src, trans in zip(src_sentences, translated_sentences):
                    # src = ' '.join(src[1:-1])
                    # trans = ' '.join(trans[:-1])
                    src = ' '.join(src)
                    trans = ' '.join(trans)
                    f.write(f"src: {src}\n")
                    f.write(f"translated: {trans}\n")
            
            with open("translated_test.txt", "a") as f:
                for trans in translated_sentences:
                    trans = ' '.join(trans[:-1])
                    f.write(trans + "\n")

            total_loss += loss.item()

    return total_loss / len(test_loader)



def calculate_bleu_score(src="translated_test.txt", tgt="ted-talks-corpus/test.fr"):
    # Read source and target files
    with io.open(src, 'r', encoding='utf-8') as f_src, io.open(tgt, 'r', encoding='utf-8') as f_tgt:
        src_sentences = f_src.readlines()
        tgt_sentences = f_tgt.readlines()

    # Ensure the files have the same number of sentences
    assert len(src_sentences) == len(tgt_sentences), "Source and target files must have the same number of sentences."
    mean_bleu_score = 0
    # Calculate BLEU scores and write to file
    with io.open('testbleu.txt', 'w', encoding='utf-8') as f_out:
        for i, (src_sent, tgt_sent) in enumerate(zip(src_sentences, tgt_sentences), 1):
            # Tokenize sentences
            src_tokens = word_tokenize(src_sent.strip(), language="french")
            tgt_tokens = word_tokenize(tgt_sent.strip(), language="french")

            # Calculate BLEU score
            bleu_score = sentence_bleu([tgt_tokens], src_tokens)
            mean_bleu_score += bleu_score
            # Write result to file
            f_out.write(f"{i} {bleu_score}\n")
    mean_bleu_score /= len(src_sentences)
    print(f"BLEU scores have been written to testbleu.txt, The mean BLEU score is : {mean_bleu_score}")


def main(args):
    # Hyperparameters (should match those used in training)
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    d_ff = args.d_ff
    max_seq_length = args.max_seq_length
    dropout = args.dropout
    batch_size = args.batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset
    test_dataset = TranslationDataset('ted-talks-corpus/test.en', 'ted-talks-corpus/test.fr', vocab_src_file = 'ted-talks-corpus/train.en', vocab_tgt_file = 'ted-talks-corpus/train.fr')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    src_vocab_size = len(test_dataset.src_vocab)
    tgt_vocab_size = len(test_dataset.tgt_vocab)

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
    model.load_state_dict(torch.load('transformer.pt'))
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # 0 is the padding index
    
    test_loss = test(model, test_loader, criterion, test_dataset.src_vocab, test_dataset.tgt_vocab, device)
    print(f'Test Loss: {test_loss:.4f}')

    # calculate_bleu_score(src="translated_test.txt", tgt="ted-talks-corpus/test.fr")
    


    
    # example_sentences = [
    #     "When I was in my 20s, I saw my very first psychotherapy client.",
    #     "I was a Ph.D. student in clinical psychology at Berkeley.",
    #     "She was a 26-year-old woman named Alex."
    # ]
    
    # for sentence in example_sentences:
    #     translation = translate(model, sentence, test_dataset.src_vocab, test_dataset.tgt_vocab, test_dataset.src_tokenizer, device)
    #     print(f"Source: {sentence}")
    #     print(f"Translation: {translation}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='test_transformers',
                description='solla mudiyadhu',
                epilog='Text at the bottom of help')
    parser.add_argument('--d_model', type=int, default=512, help='Dimensionality of the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the model')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimensionality of the feed-forward layer')
    parser.add_argument('--max_seq_length', type=int, default=100, help='Maximum sequence length for input')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--wandb', type=bool, default=False, help='Use Weights & Biases for logging')


    args = parser.parse_args()

    main(args)