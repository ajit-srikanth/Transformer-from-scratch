import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from encoder import Encoder
from decoder import Decoder
from utils import TranslationDataset, collate_batch, create_mask
import wandb
from tqdm import tqdm
import argparse

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        return output

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    losses = []
    for batch in tqdm(train_loader):
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        
        src_mask, tgt_mask = create_mask(src, tgt)
        
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :, :])
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        losses.append(loss.item())
    return total_loss / len(train_loader), losses

def main(args):
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    d_ff = args.d_ff
    max_seq_length = args.max_seq_length
    dropout = args.dropout
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    
    wandb_ah = args.wandb
    
    # Initialize wandb
    if wandb_ah:
        wandb.init(project="transformer-translation", name="en-fr-translation")
    
    # Log hyperparameters
    if wandb_ah:
        wandb.config.update({
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "max_seq_length": max_seq_length,
            "dropout": dropout,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate
        })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_dataset = TranslationDataset('ted-talks-corpus/train.en', 'ted-talks-corpus/train.fr')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    print("need to optimise loading twice (for val...building same vocab again) lol")
    val_dataset = TranslationDataset('ted-talks-corpus/dev.en', 'ted-talks-corpus/dev.fr', vocab_src_file = 'ted-talks-corpus/train.en', vocab_tgt_file = 'ted-talks-corpus/train.fr')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


    src_vocab_size = len(train_dataset.src_vocab)   
    tgt_vocab_size = len(train_dataset.tgt_vocab)
    
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
    
    if wandb_ah:
        wandb.watch(model)  # Log model architecture
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is the padding index
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    for epoch in tqdm(range(num_epochs)):
        train_loss, batch_losses = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')

        with torch.no_grad():
            model.eval()
            total_loss = 0
            losses = []
            for batch in tqdm(val_loader):
                src, tgt = batch
                src, tgt = src.to(device), tgt.to(device)
                
                src_mask, tgt_mask = create_mask(src, tgt)
        
                output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :, :])
                loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
                           
                total_loss += loss.item()
                losses.append(loss.item())
                val_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch+1}, Validation Loss: {val_loss:.4f}')
        # Log training loss
        if wandb_ah:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "batch_losses": wandb.Histogram(batch_losses),
                "val_loss": val_loss
            })
    
    torch.save(model.state_dict(), 'transformer.pt')
    # wandb.save('transformer.pt')  # Save the model file to wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='train_transformers',
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