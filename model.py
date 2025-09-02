from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F      




# 📤 Encoder: GRU ساده
class Encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim,hidden_dim,num_layers,batch_first=True)
        
        
    def forward(self,x):
        outputs,hidden = self.gru(x)
        return outputs , hidden 
    
    
    

# 🎯 Attention Layer
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [1, B, H], encoder_outputs: [B, T_in, H]
        T = encoder_outputs.shape[1]
        decoder_hidden = decoder_hidden.permute(1, 0, 2)  # [B, 1, H]
        decoder_hidden = decoder_hidden.repeat(1, T, 1)   # [B, T, H]
        
        
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden,encoder_outputs),dim=2)))
        scores = self.v(energy).squeeze(2)
        attn_weights = F.softmax(scores,dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1),encoder_outputs)
        
        
        return context
    
    

# 📥 Decoder با Attention

class Decoder(nn.Module):
    def __init__(self,hidden_dim,output_dim=1,num_layers=1):
        super().__init__()
        self.gru = nn.GRU(hidden_dim+1,hidden_dim,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)
        self.attention = Attention(hidden_dim)
        
        
    
    def forward(self, x, hidden, encoder_outputs):
        # x: [B, 1, 1]
        context = self.attention(hidden, encoder_outputs)  # [B, 1, H]
        rnn_input = torch.cat((x, context), dim=2)  # [B, 1, H+1]
        output, hidden = self.gru(rnn_input, hidden)  # output: [B, 1, H]
        prediction = self.fc(output)  # [B, 1, 1]
        return prediction, hidden
    
    

# 🔁 Seq2Seq Full Model
class Seq2Seq(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_window=72):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim)
        self.output_window = output_window

    def forward(self, src):
        # src: [B, T_in]
        src = src.unsqueeze(2)  # → [B, T_in, 1]
        encoder_outputs, hidden = self.encoder(src)

        decoder_input = src[:, -1:, :]  # آخرین دمای دیده‌شده
        outputs = []

        for _ in range(self.output_window):
            pred, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs.append(pred)
            decoder_input = pred.detach()  # teacher forcing: off (inference style)

        return torch.cat(outputs, dim=1).squeeze(2)  # [B, output_window]