import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for temporal sequences
    """
    def __init__(self, input_dim=771, hidden_dim=128, num_layers=2, 
                 num_classes=3, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * num_directions)
        
        # Take the last output
        if self.bidirectional:
            # Concatenate last hidden states from both directions
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Classification
        output = self.fc(hidden)
        
        return output


class FCNClassifier(nn.Module):
    """
    Fully Convolutional Network (FCN) for time series classification
    Similar to the one used in DSTS paper
    """
    def __init__(self, input_dim=771, num_classes=3):
        super(FCNClassifier, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, 
                               kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, 
                               kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        
        # Conv Block 3
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, 
                               kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Transpose to (batch_size, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global Average Pooling
        x = self.gap(x)  # (batch_size, 128, 1)
        x = x.squeeze(-1)  # (batch_size, 128)
        
        # Classification
        output = self.fc(x)
        
        return output


class InceptionModule(nn.Module):
    """
    Inception module for time series
    """
    def __init__(self, in_channels, nb_filters=32):
        super(InceptionModule, self).__init__()
        
        self.bottleneck = nn.Conv1d(in_channels, nb_filters, kernel_size=1, padding='same')
        
        # Different kernel sizes
        self.conv1 = nn.Conv1d(nb_filters, nb_filters, kernel_size=39, padding='same')
        self.conv2 = nn.Conv1d(nb_filters, nb_filters, kernel_size=19, padding='same')
        self.conv3 = nn.Conv1d(nb_filters, nb_filters, kernel_size=9, padding='same')
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, nb_filters, kernel_size=1, padding='same')
        
        self.bn = nn.BatchNorm1d(nb_filters * 4)
        
    def forward(self, x):
        # Bottleneck
        bottleneck = self.bottleneck(x)
        
        # Parallel convolutions
        conv1 = self.conv1(bottleneck)
        conv2 = self.conv2(bottleneck)
        conv3 = self.conv3(bottleneck)
        
        # Max pooling path
        pool = self.maxpool(x)
        conv_pool = self.conv_pool(pool)
        
        # Concatenate
        out = torch.cat([conv1, conv2, conv3, conv_pool], dim=1)
        out = self.bn(out)
        out = F.relu(out)
        
        return out


class InceptionTimeClassifier(nn.Module):
    """
    InceptionTime classifier (simplified version)
    Similar to the one used in DSTS paper
    """
    def __init__(self, input_dim=771, num_classes=3, depth=6, nb_filters=32):
        super(InceptionTimeClassifier, self).__init__()
        
        self.depth = depth
        
        # Initial conv
        self.initial_conv = nn.Conv1d(input_dim, nb_filters, kernel_size=1)
        
        # Inception modules
        self.inception_modules = nn.ModuleList()
        current_channels = nb_filters
        
        for i in range(depth):
            self.inception_modules.append(
                InceptionModule(current_channels, nb_filters)
            )
            current_channels = nb_filters * 4  # After concatenation
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.fc = nn.Linear(current_channels, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Transpose to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Initial conv
        x = self.initial_conv(x)
        
        # Inception modules
        for inception in self.inception_modules:
            x = inception(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.squeeze(-1)
        
        # Classification
        output = self.fc(x)
        
        return output


class MCNNClassifier(nn.Module):
    """
    Multi-scale CNN (MCNN) for time series classification
    Similar to the one used in DSTS paper
    """
    def __init__(self, input_dim=771, num_classes=3):
        super(MCNNClassifier, self).__init__()
        
        # Multi-scale convolutions
        self.conv_small = nn.Conv1d(input_dim, 64, kernel_size=3, padding='same')
        self.conv_medium = nn.Conv1d(input_dim, 64, kernel_size=7, padding='same')
        self.conv_large = nn.Conv1d(input_dim, 64, kernel_size=11, padding='same')
        
        self.bn1 = nn.BatchNorm1d(192)  # 64 * 3
        
        # Second layer
        self.conv2 = nn.Conv1d(192, 128, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # Output layer
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Transpose to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Multi-scale convolutions
        x_small = self.conv_small(x)
        x_medium = self.conv_medium(x)
        x_large = self.conv_large(x)
        
        # Concatenate
        x = torch.cat([x_small, x_medium, x_large], dim=1)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second conv
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.squeeze(-1)
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        output = self.fc(x)
        
        return output


def get_model(model_name, input_dim, num_classes, **kwargs):
    """
    Factory function to get model by name
    
    Args:
        model_name: str, one of ['lstm', 'fcn', 'inception', 'mcnn']
        input_dim: int, feature dimension (771)
        num_classes: int, number of classes
        **kwargs: additional model-specific arguments
    
    Returns:
        model: nn.Module
    """
    model_name = model_name.lower()
    
    if model_name == 'lstm':
        return LSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.3),
            bidirectional=kwargs.get('bidirectional', True)
        )
    
    elif model_name == 'fcn':
        return FCNClassifier(
            input_dim=input_dim,
            num_classes=num_classes
        )
    
    elif model_name == 'inception':
        return InceptionTimeClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            depth=kwargs.get('depth', 6),
            nb_filters=kwargs.get('nb_filters', 32)
        )
    
    elif model_name == 'mcnn':
        return MCNNClassifier(
            input_dim=input_dim,
            num_classes=num_classes
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: lstm, fcn, inception, mcnn")


if __name__ == "__main__":
    # Test models
    batch_size = 16
    seq_len = 100
    input_dim = 771
    num_classes = 3
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("Testing models with input shape:", x.shape)
    print("=" * 60)
    
    for model_name in ['lstm', 'fcn', 'inception', 'mcnn']:
        print(f"\n{model_name.upper()}:")
        model = get_model(model_name, input_dim, num_classes)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
        
        # Forward pass
        output = model(x)
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({batch_size}, {num_classes})")
        
        assert output.shape == (batch_size, num_classes), "Output shape mismatch!"
    
    print("\n✅ All models working correctly!")