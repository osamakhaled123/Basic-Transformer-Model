import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

class positional_encoding(nn.Module):
    def __init__(self, max_length, emb_dim):
        super(positional_encoding, self).__init__()

        self.pos_enc = torch.zeros(size=(max_length, emb_dim))
        self.pos=torch.arange(1, max_length+1, dtype=torch.float32).unsqueeze(1).repeat(1,emb_dim)
        self.equation = torch.pow(max_length, (2*torch.arange(emb_dim))/emb_dim).tile(max_length,1)

    def forward(self):
        self.pos_enc[:,0::2] = torch.sin(self.pos[:,0::2] / self.equation[:,0::2])
        self.pos_enc[:,1::2] = torch.cos(self.pos[:,1::2] / self.equation[:,1::2])

        pos_vector = self.pos_enc.unsqueeze(0)
       
        return pos_vector
    


class scaled_dot_product_attention(nn.Module):
    def __init__(self, emb_dim, causal = False, dropout = 0.1): #size=(batch, max_len, emb_dim)
        super(scaled_dot_product_attention, self).__init__()

        self.Q_linear = nn.Linear(emb_dim, emb_dim, dtype=torch.float32)
        self.K_linear = nn.Linear(emb_dim, emb_dim, dtype=torch.float32)
        self.V_linear = nn.Linear(emb_dim, emb_dim, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attention_mask):
        Q = self.Q_linear(Q)
        Q = self.dropout(Q)

        K = self.K_linear(K)
        K = self.dropout(K)

        V = self.V_linear(V)
        V = self.dropout(V)

        scores = torch.matmul(Q, torch.transpose(K,-1,-2))
        dk = torch.sqrt(torch.tensor(K.size(-1)))
        scores /= dk
        #print(f"Scores size is {scores.shape} with Q size {Q.shape} and K size {K.shape}",end='\n\n')
        #print(f"attention mask shape is {attention_mask.shape}", end='\n\n')
        scores = scores.masked_fill(attention_mask == 0, float(-1e10))

        if self.causal:
            mask = torch.ones(size=(scores.size(-1), scores.size(-2))).to(scores.device)
            mask = torch.triu(input=mask, diagonal=1)
            mask = mask.masked_fill(mask == 1, float(-1e10))
            scores += mask

        attention_weights = self.softmax(scores)
        results = torch.matmul(attention_weights, V)

        return results    
    


class multihead_attention(nn.Module):
    def __init__(self, num_heads, emb_dim, causal=False, dropout=0.1):
        super(multihead_attention, self).__init__()

        self.input_batch = int(emb_dim // num_heads)
        self.linear_projection = nn.Linear(emb_dim, emb_dim, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            scaled_dot_product_attention(self.input_batch, causal, dropout)
            for head in range(num_heads)])

    def forward(self, Q, K, V, attention_mask):
        outputs = []
        range = 0
        for head in self.heads:
            outputs.append(head(Q[:,:,range:range+self.input_batch],
                                K[:,:,range:range+self.input_batch],
                                V[:,:,range:range+self.input_batch],
                                attention_mask))
            range += self.input_batch

        concatenated_heads_outputs = torch.cat(outputs, dim=-1)
        linear_projection = self.linear_projection(concatenated_heads_outputs)
        linear_projection = self.dropout(linear_projection)

        return linear_projection



class transformer_encoder_decoder(nn.Module):
    def __init__(self, num_heads, emb_dim, dff, dropout=0.1):
        super(transformer_encoder_decoder, self).__init__()
        self.multi_head_attention = multihead_attention(num_heads, emb_dim,
                                                        dropout=dropout)
        self.layer_normalization_1 = nn.LayerNorm(emb_dim, dtype=torch.float32)
        self.layer_normalization_2 = nn.LayerNorm(emb_dim, dtype=torch.float32)
        self.RelU_layer = nn.Linear(emb_dim, dff, dtype=torch.float32)
        self.RelU = nn.ReLU()
        self.Linear_layer = nn.Linear(dff, emb_dim, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attention_mask):
        layer = self.multi_head_attention(Q, K, V, attention_mask)
        layer += Q
        normalized = self.layer_normalization_1(layer)
        #The dimensionality of input and output is dmodel = 512,
        #and the inner-layer has dimensionality dff = 2048
        layer = self.RelU_layer(normalized)
        layer = self.RelU(layer)
        layer = self.dropout(layer)
        layer = self.Linear_layer(layer)
        layer = self.dropout(layer)
        layer += normalized
        output = self.layer_normalization_2(layer)

        return output



class Transformer(nn.Module):
    def __init__(self, vocab, max_input_length, max_target_length,
                 emb_dim, dff, num_heads, num_encoder_blocks, num_decoder_blocks,
                 dropout):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.dff = dff
        self.Dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(len(vocab), emb_dim, padding_idx=vocab['<pad>'])
        self.positional_encoding_encoder = positional_encoding(max_input_length-1, emb_dim)
        self.positional_encoding_decoder = positional_encoding(max_target_length-1, emb_dim)

        self.encoder_blocks = nn.ModuleList([transformer_encoder_decoder(num_heads,
        emb_dim, dff, dropout) for _ in range(num_encoder_blocks)])

        self.decoder_blocks = nn.ModuleList([transformer_encoder_decoder(num_heads,
        emb_dim, dff, dropout) for _ in range(num_decoder_blocks)])

        self.masked_attention = multihead_attention(num_heads, emb_dim, True, dropout)
        self.layer_normalization = nn.LayerNorm(emb_dim, dtype=torch.float32)

    def forward(self, batch, target):
        #Encoder Part
        training_data = self.embedding(batch)
        #Positional Encoding masking for training data
        positional_mask_batch = (batch != self.vocab['<pad>']).unsqueeze(-1).float()
        training_data *= positional_mask_batch

        # Slice positional encoding to match the sequence length of the batch
        seq_len_batch = training_data.size(-2)
        training_data += self.positional_encoding_encoder().to(training_data.device)[:, :seq_len_batch, :] * positional_mask_batch
        #Attention masking for training data
        attention_mask_batch = (batch != self.vocab['<pad>']).unsqueeze(-1).float()

        encoder_output = self.encoder_decoder(training_data,
                                              training_data,
                                              training_data,
                                               self.encoder_blocks,
                                               attention_mask_batch)
        K, V = encoder_output, encoder_output

        #Decoder Part
        target_data = self.embedding(target)
        #Positional Encoding masking for target data
        positional_mask_target = (target != self.vocab['<pad>']).unsqueeze(-1).float()
        target_data *= positional_mask_target
        #print(target_data.size())
        # Slice positional encoding to match the sequence length of the target
        seq_len_target = target_data.size(-2)
        #print(self.positional_encoding_decoder().to(target_data.device)[:, :seq_len_target, :].size())
        target_data += self.positional_encoding_decoder().to(target_data.device)[:, :seq_len_target, :] * positional_mask_target
        #Attention masking for target data
        attention_mask_target = (target != self.vocab['<pad>']).unsqueeze(-1).float()

        #print(f"target data size after embedding and masking is\t{target_data.shape}")
        masked_attention_output = self.masked_attention(target_data,
                                                        target_data,
                                                        target_data,
                                                        attention_mask_target)
        out = masked_attention_output + target_data
        Q = self.layer_normalization(out)

        decoder_output = self.encoder_decoder(Q, K, V, self.decoder_blocks,
                                              attention_mask_target)

        pre_softmax = torch.nn.functional.linear(decoder_output, self.embedding.weight)
        pre_softmax = self.Dropout(pre_softmax)

        return pre_softmax
    #Encoder-Decoder Blocks
    def encoder_decoder(self, Q, K, V, blocks, attention_mask):
        for block in blocks:
            result = block(Q, K, V, attention_mask)
            Q, K, V = result, result, result

        return result
    

def train(model, train_set, val_set, epochs, lr, device):
    criterion = nn.CrossEntropyLoss(ignore_index=model.vocab['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_accuracy = 0.0
        val_accuracy = 0.0
        training_loss = 0.0
        validation_loss = 0.0

        model.train()
        train_loop = tqdm(train_set, desc=f"Train Epoch {epoch+1}", leave=False)

        for train_batch, target_train_batch in train_loop:
            optimizer.zero_grad()

            train_batch, target_train_batch = train_batch.to(device), target_train_batch.to(device)

            predicted_output = model(train_batch[:,1:], target_train_batch[:,:-1])

            logits = predicted_output.contiguous().view(-1, predicted_output.size(-1))
            targets = target_train_batch[:,1:].contiguous().view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            #_, pred_class = torch.max(predicted_output, 1)
            #equals = pred_class == target_train_batch.view(*pred_class.shape)
            #train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
            preds = torch.argmax(predicted_output, dim=-1)
            #print(f"preds shape is\t{preds.shape}")
            #print(f"targets shape is\t{targets.shape}")
            mask = (targets != model.vocab['<pad>'])
            accuracy = ((preds.view(-1) == targets) & mask).sum() / mask.sum()
            train_accuracy += accuracy

        train_losses.append(training_loss/len(train_set))
        print(f"Epoch {epoch+1} accumelated Training batches loss is:\t\t {training_loss/len(train_set)}")
        print(f"Epoch {epoch+1} accuracy on Training batches is:\t\t {train_accuracy / len(train_set)}")


        model.eval()
        val_loop = tqdm(val_set, desc=f"Validation Epoch {epoch+1}", leave=False)

        with torch.no_grad():
            for valid_batch, target_valid_batch in val_loop:

                valid_batch, target_valid_batch = valid_batch.to(device), target_valid_batch.to(device)
                predicted_output = model(valid_batch[:,1:], target_valid_batch[:,:-1])

                logits = predicted_output.contiguous().view(-1, predicted_output.size(-1)).clone().detach()
                targets = target_valid_batch[:,1:].contiguous().view(-1).clone().detach()

                loss = criterion(logits, targets)
                validation_loss += loss.item()
                #_, pred_class=torch.max(predicted_output, 1)
                #equals = pred_class == target_valid_batch.view(*pred_class.shape)
                #val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                preds = torch.argmax(predicted_output, dim=-1)
                mask = (targets != model.vocab['<pad>'])
                accuracy = ((preds.view(-1) == targets) & mask).sum() / mask.sum()
                val_accuracy += accuracy

        val_losses.append(validation_loss/len(val_set))
        print(f"Epoch {epoch+1} accumelated validation batches loss is:\t\t {validation_loss/len(val_set)}")
        print(f"Epoch {epoch+1} accuracy on validation batches is:\t\t {val_accuracy / len(val_set)}")

    return train_losses, val_losses
