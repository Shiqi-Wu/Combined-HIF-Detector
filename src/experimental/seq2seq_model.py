import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_size, encoder_output_size):
        super().__init__()
        self.attn_linear = nn.Linear(hidden_size + encoder_output_size, hidden_size)
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)
        self.output_proj = nn.Linear(hidden_size + encoder_output_size, hidden_size)

    def forward(self, decoder_output, encoder_outputs):
        """
        decoder_output: [B, 1, H]
        encoder_outputs: [B, T, H_enc]
        """
        T = encoder_outputs.size(1)
        decoder_exp = decoder_output.repeat(1, T, 1)  # [B, T, H]
        combined = torch.cat([decoder_exp, encoder_outputs], dim=-1)  # [B, T, H+H_enc]
        energy = torch.tanh(self.attn_linear(combined))
        attn_weights = F.softmax(self.attn_score(energy), dim=1)  # [B, T, 1]
        context = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)  # [B, 1, H_enc]
        attended = torch.cat([decoder_output, context], dim=-1)
        return self.output_proj(attended)  # [B, 1, H]

class Seq2SeqLSTM(nn.Module):
    def __init__(self, state_dim, control_dim, hidden_size=128,
                 num_layers=1, dropout=0.0, bidirectional=True,
                 use_attention=True, teacher_forcing_ratio=0.5):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_attention = use_attention
        self.bidirectional = bidirectional

        self.encoder = nn.LSTM(state_dim, hidden_size, num_layers=num_layers,
                               dropout=dropout if num_layers > 1 else 0,
                               batch_first=True, bidirectional=bidirectional)

        encoder_output_size = hidden_size * (2 if bidirectional else 1)
        self.decoder = nn.LSTM(control_dim + encoder_output_size,
                               hidden_size, batch_first=True)

        if use_attention:
            self.attention = AttentionMechanism(hidden_size, encoder_output_size)

        self.output_layer = nn.Linear(hidden_size, control_dim)

    def forward(self, x, u=None, max_length=None):
        """
        x: [B, T, state_dim]
        u: [B, T, control_dim] or None
        """
        B, T, _ = x.shape
        max_length = max_length or T
        device = x.device

        encoder_outputs, (h, c) = self.encoder(x)
        if self.bidirectional:
            h = (h[0:h.size(0):2] + h[1:h.size(0):2]) / 2
            c = (c[0:c.size(0):2] + c[1:c.size(0):2]) / 2

        decoder_hidden = (h, c)
        predictions = torch.zeros(B, max_length, self.control_dim, device=device)
        prev_u = torch.zeros(B, 1, self.control_dim, device=device)

        for t in range(max_length):
            if t < T:
                context = encoder_outputs[:, t:t+1, :]
            else:
                context = encoder_outputs[:, -1:, :]

            decoder_input = torch.cat([prev_u, context], dim=-1)
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            if self.use_attention:
                output = self.attention(output, encoder_outputs)

            u_t = self.output_layer(output)
            predictions[:, t:t+1, :] = u_t

            if self.training and u is not None and t < T and torch.rand(1).item() < self.teacher_forcing_ratio:
                prev_u = u[:, t:t+1, :]
            else:
                prev_u = u_t

        return predictions

    def predict_control(self, x, output_length=None, temperature=1.0, return_last=False):
        """
        x: [B, T, state_dim]
        return_last: if True, returns only final u(t)
        """
        self.eval()
        with torch.no_grad():
            out_len = output_length or x.size(1)
            preds = self.forward(x, u=None, max_length=out_len)
            if temperature != 1.0:
                preds = preds / temperature
            return preds[:, -1:, :] if return_last else preds
