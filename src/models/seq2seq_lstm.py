"""
Sequence-to-Sequence LSTM model for u(t) generation
Simple approach: u(t) depends on x(0:t) and u(0:t-1) through encoder-decoder architecture

Model Architecture:
- Encoder: Processes PCA-reduced state sequences x(0:t) to capture temporal dependencies
- Decoder: Generates control sequences u(t) step by step
- Teacher forcing during training, autoregressive during inference
- Direct mapping from state history to control output

Key Dependencies:
- u(t) = f(x(0:t), u(0:t-1))
- x(t) is PCA-processed state (typically 2D after dimensionality reduction)
- Encoder provides rich representation of state history
- Decoder generates control signals autoregressively
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class Seq2SeqLSTM(nn.Module):
    """
    Sequence-to-Sequence LSTM for control signal generation
    
    Architecture:
    - Encoder LSTM: Processes state signals x(0:t) to capture temporal patterns
    - Decoder LSTM: Generates control signals u(t) autoregressively
    - Teacher forcing during training, autoregressive during inference
    
    Simple approach: u(t) = f(x(0:t), u(0:t-1))
    """
    
    def __init__(self, 
                 state_dim: int,
                 control_dim: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 use_attention: bool = False,
                 teacher_forcing_ratio: float = 0.5):
        """
        Initialize Seq2Seq LSTM model
        
        Args:
            state_dim: Dimension of PCA-processed state signals x(t) (typically 2)
            control_dim: Dimension of control signals u(t)
            hidden_size: Hidden size for LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM in encoder
            use_attention: Whether to use attention mechanism
            teacher_forcing_ratio: Probability of using teacher forcing during training
        """
        super(Seq2SeqLSTM, self).__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Encoder LSTM - processes state sequences x(0:t)
        self.encoder = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Calculate encoder output size
        encoder_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Decoder LSTM - generates control sequences u(t)
        # Input: [u(t-1), encoder_context_at_t]
        decoder_input_size = control_dim + encoder_output_size
        self.decoder = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Decoder is always unidirectional
        )
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = AttentionMechanism(hidden_size, encoder_output_size)
        
        # Output projection layer
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, control_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Xavier initialization for LSTM weights
                    nn.init.xavier_uniform_(param.data)
                else:
                    # Normal initialization for linear layers
                    nn.init.normal_(param.data, 0, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def encode(self, state_sequences: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode state sequences x(0:t) to capture temporal dependencies
        
        Args:
            state_sequences: [batch_size, seq_len, state_dim]
            
        Returns:
            encoder_outputs: [batch_size, seq_len, encoder_output_size] - context for each time step
            encoder_hidden: (h_n, c_n) final hidden states
        """
        encoder_outputs, encoder_hidden = self.encoder(state_sequences)
        return encoder_outputs, encoder_hidden
    
    def decode_step(self, 
                   previous_control: torch.Tensor,
                   encoder_context: torch.Tensor,
                   decoder_hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single decoding step: generate u(t) from u(t-1) and encoder context
        
        Args:
            previous_control: [batch_size, 1, control_dim] - u(t-1)
            encoder_context: [batch_size, 1, encoder_output_size] - context from x(0:t)
            decoder_hidden: Previous decoder hidden state
            
        Returns:
            output: [batch_size, 1, control_dim] - u(t)
            new_hidden: Updated decoder hidden state
        """
        # Concatenate u(t-1) and encoder context
        decoder_input = torch.cat([previous_control, encoder_context], dim=-1)
        
        # LSTM forward pass
        decoder_output, new_hidden = self.decoder(decoder_input, decoder_hidden)
        
        # Apply attention if enabled
        if self.use_attention:
            decoder_output = self.attention(decoder_output, encoder_context)
        
        # Project to control dimension
        output = self.output_projection(decoder_output)
        
        return output, new_hidden
    
    def forward(self, 
                state_sequences: torch.Tensor,
                control_sequences: Optional[torch.Tensor] = None,
                max_length: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass: generate u(t) from x(0:t) and u(0:t-1)
        
        Args:
            state_sequences: [batch_size, seq_len, state_dim] - x(0:T)
            control_sequences: [batch_size, seq_len, control_dim] - u(0:T) (for teacher forcing)
            max_length: Maximum generation length (for inference)
            
        Returns:
            predictions: [batch_size, seq_len, control_dim] - predicted u(0:T)
        """
        batch_size, seq_len, _ = state_sequences.shape
        device = state_sequences.device
        
        # Encode state sequences to get context for each time step
        encoder_outputs, encoder_hidden = self.encode(state_sequences)
        
        # Initialize decoder hidden state from encoder
        decoder_hidden = self._init_decoder_hidden(encoder_hidden)
        
        # Prepare output tensor
        if max_length is None:
            max_length = seq_len
        
        predictions = torch.zeros(batch_size, max_length, self.control_dim, device=device)
        
        # Initial control signal u(-1) = 0 (no previous control)
        previous_control = torch.zeros(batch_size, 1, self.control_dim, device=device)
        
        # Generate sequence step by step
        for t in range(max_length):
            # Get encoder context for time t
            if t < seq_len:
                encoder_context = encoder_outputs[:, t:t+1, :]  # [batch_size, 1, encoder_output_size]
            else:
                # For inference beyond training sequence length, use last context
                encoder_context = encoder_outputs[:, -1:, :]
            
            # Decode step: u(t) = f(u(t-1), context_from_x(0:t))
            output, decoder_hidden = self.decode_step(
                previous_control=previous_control,
                encoder_context=encoder_context,
                decoder_hidden=decoder_hidden
            )
            
            predictions[:, t:t+1, :] = output
            
            # Prepare next input: u(t-1) for next step becomes current u(t)
            if self.training and control_sequences is not None and t < seq_len - 1:
                # Teacher forcing: use ground truth or predicted output for next step
                use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
                if use_teacher_forcing:
                    previous_control = control_sequences[:, t:t+1, :]  # Use ground truth u(t)
                else:
                    previous_control = output  # Use predicted u(t)
            else:
                previous_control = output  # Use predicted u(t)
        
        return predictions
    
    def _init_decoder_hidden(self, encoder_hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize decoder hidden state from encoder hidden state
        
        Args:
            encoder_hidden: (h_n, c_n) from encoder
            
        Returns:
            decoder_hidden: (h_0, c_0) for decoder
        """
        h_n, c_n = encoder_hidden
        
        if self.bidirectional:
            # For bidirectional encoder, combine forward and backward hidden states
            h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
            c_n = c_n.view(self.num_layers, 2, -1, self.hidden_size)
            
            # Combine forward and backward states
            h_n = (h_n[:, 0, :, :] + h_n[:, 1, :, :]) / 2
            c_n = (c_n[:, 0, :, :] + c_n[:, 1, :, :]) / 2
        
        return h_n, c_n
    
    def generate_sequence(self, 
                         state_sequences: torch.Tensor,
                         max_length: int,
                         temperature: float = 1.0) -> torch.Tensor:
        """
        Generate control sequence without teacher forcing (inference mode)
        
        Args:
            state_sequences: [batch_size, seq_len, state_dim]
            max_length: Maximum generation length
            temperature: Sampling temperature for diversity
            
        Returns:
            generated_sequences: [batch_size, max_length, control_dim]
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(state_sequences, max_length=max_length)
            
            if temperature != 1.0:
                predictions = predictions / temperature
            
            return predictions


class AttentionMechanism(nn.Module):
    """
    Attention mechanism for seq2seq model
    """
    
    def __init__(self, hidden_size: int, encoder_output_size: int):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_output_size = encoder_output_size
        
        # Attention layers
        self.attention_linear = nn.Linear(hidden_size + encoder_output_size, hidden_size)
        self.attention_v = nn.Linear(hidden_size, 1, bias=False)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size + encoder_output_size, hidden_size)
    
    def forward(self, 
                decoder_output: torch.Tensor,
                encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism
        
        Args:
            decoder_output: [batch_size, 1, hidden_size]
            encoder_outputs: [batch_size, seq_len, encoder_output_size]
            
        Returns:
            attended_output: [batch_size, 1, hidden_size]
        """
        seq_len = encoder_outputs.size(1)
        
        # Repeat decoder output for each encoder position
        decoder_output_repeated = decoder_output.repeat(1, seq_len, 1)
        
        # Calculate attention weights
        attention_input = torch.cat([decoder_output_repeated, encoder_outputs], dim=-1)
        attention_hidden = torch.tanh(self.attention_linear(attention_input))
        attention_weights = F.softmax(self.attention_v(attention_hidden), dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1, keepdim=True)
        
        # Combine context with decoder output
        combined = torch.cat([decoder_output, context_vector], dim=-1)
        attended_output = self.output_projection(combined)
        
        return attended_output


class Seq2SeqLoss(nn.Module):
    """
    Custom loss function for seq2seq training
    Combines MSE loss with optional regularization terms
    """
    
    def __init__(self, 
                 mse_weight: float = 1.0,
                 l1_weight: float = 0.0,
                 smoothness_weight: float = 0.0):
        super(Seq2SeqLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.smoothness_weight = smoothness_weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss
        
        Args:
            predictions: [batch_size, seq_len, control_dim]
            targets: [batch_size, seq_len, control_dim]
            
        Returns:
            total_loss: Combined loss value
        """
        total_loss = 0.0
        
        # MSE loss
        if self.mse_weight > 0:
            mse_loss = self.mse_loss(predictions, targets)
            total_loss += self.mse_weight * mse_loss
        
        # L1 loss
        if self.l1_weight > 0:
            l1_loss = self.l1_loss(predictions, targets)
            total_loss += self.l1_weight * l1_loss
        
        # Smoothness loss (penalize large changes between consecutive time steps)
        if self.smoothness_weight > 0:
            pred_diff = predictions[:, 1:, :] - predictions[:, :-1, :]
            target_diff = targets[:, 1:, :] - targets[:, :-1, :]
            smoothness_loss = self.mse_loss(pred_diff, target_diff)
            total_loss += self.smoothness_weight * smoothness_loss
        
        return total_loss


# Utility functions for model analysis
def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_model_complexity(model: Seq2SeqLSTM) -> dict:
    """Analyze model complexity and provide statistics"""
    stats = {
        'total_parameters': count_parameters(model),
        'encoder_parameters': count_parameters(model.encoder),
        'decoder_parameters': count_parameters(model.decoder),
        'output_projection_parameters': count_parameters(model.output_projection),
        'model_size_mb': count_parameters(model) * 4 / (1024 * 1024),  # Assuming float32
    }
    
    if model.use_attention:
        stats['attention_parameters'] = count_parameters(model.attention)
    
    return stats


if __name__ == "__main__":
    # Test model instantiation
    model = Seq2SeqLSTM(
        state_dim=2,  # PCA-processed state dimension
        control_dim=2,  # Control signal dimension (from data columns -6:-4)
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        use_attention=True,
        teacher_forcing_ratio=0.5
    )
    
    print("Simple Seq2Seq Model Architecture:")
    print("- u(t) depends on x(0:t) and u(0:t-1)")
    print("- Encoder processes x(0:T) to provide context")
    print("- Decoder generates u(t) using [u(t-1), encoder_context]")
    print("- Teacher forcing during training")
    print("\nModel Structure:")
    print(model)
    
    print("\nModel Complexity Analysis:")
    stats = analyze_model_complexity(model)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test forward pass
    batch_size, seq_len, state_dim, control_dim = 2, 30, 2, 2  # Both state and control are 2D
    
    x = torch.randn(batch_size, seq_len, state_dim)  # State sequences x(0:T) - PCA processed
    u = torch.randn(batch_size, seq_len, control_dim)  # Control sequences u(0:T) - 2D control signals
    
    # Training mode (with teacher forcing)
    model.train()
    predictions = model(x, u)
    print(f"\nTraining mode:")
    print(f"Input state sequences x(0:T): {x.shape}")
    print(f"Input control sequences u(0:T): {u.shape}")
    print(f"Predicted control sequences: {predictions.shape}")
    
    # Inference mode
    model.eval()
    generated = model.generate_sequence(x, max_length=seq_len)
    print(f"\nInference mode:")
    print(f"Input state sequences x(0:T): {x.shape}")
    print(f"Generated control sequences: {generated.shape}")
    
    # Test loss
    criterion = Seq2SeqLoss(mse_weight=1.0, l1_weight=0.1, smoothness_weight=0.01)
    loss = criterion(predictions, u)
    print(f"\nLoss: {loss.item()}")
    
    # Demonstrate the simple dependency
    print(f"\nSimple dependency demonstration:")
    print(f"For each time step t:")
    print(f"1. Encoder processes x(0:T) to get context for each time step")
    print(f"2. At time t, decoder uses u(t-1) and context from x(0:t)")
    print(f"3. Output: u(t) = f(u(t-1), context_from_x(0:t))")
    print(f"4. This naturally captures the dependency u(t) on x(0:t) and u(0:t-1)")
