#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MISO - Q-LOGIK Neural RNN

Implementierung fortschrittlicher RNN-Architekturen für Q-LOGIK.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import threading

# Importiere PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importiere Basisklassen
from miso.logic.qlogik_neural_base import BaseModel, RNNBase

# Importiere GPU-Beschleunigung
from miso.logic.qlogik_gpu_acceleration import (
    to_tensor, to_numpy, matmul, attention, parallel_map, batch_process
)

# Importiere Speicheroptimierung
from miso.logic.qlogik_memory_optimization import (
    get_from_cache, put_in_cache, clear_cache, register_lazy_loader,
    checkpoint, checkpoint_function
)

# Logger einrichten
logger = logging.getLogger("MISO.Logic.Q-LOGIK.NeuralRNN")

class LSTMModel(RNNBase):
    """LSTM-Implementierung für Q-LOGIK"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert das LSTM-Modell
        
        Args:
            config: Konfigurationsobjekt für das Modell
        """
        super().__init__(config)
        
        # LSTM-spezifische Konfiguration
        self.attention_enabled = self.config.get("attention_enabled", False)
        
        # Wenn Attention aktiviert ist, füge Attention-Layer hinzu
        if self.attention_enabled:
            attention_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
            self.attention = nn.Linear(attention_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das LSTM-Modell
        
        Args:
            x: Eingabetensor [batch_size, seq_len, input_size]
            
        Returns:
            Ausgabetensor [batch_size, num_classes]
        """
        # LSTM-Layer
        output, (hidden, cell) = self.rnn(x)
        
        if self.attention_enabled:
            # Attention-Mechanismus
            attention_weights = F.softmax(self.attention(output).squeeze(-1), dim=1)
            context_vector = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)
            return self.fc(context_vector)
        else:
            # Standard-Implementierung: Verwende letzten Hidden-State
            if self.bidirectional:
                # Konkateniere vorwärts und rückwärts
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            else:
                hidden = hidden[-1, :, :]
            
            return self.fc(hidden)


class GRUModel(RNNBase):
    """GRU-Implementierung für Q-LOGIK"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert das GRU-Modell
        
        Args:
            config: Konfigurationsobjekt für das Modell
        """
        # Überschreibe RNN-Typ in der Konfiguration
        config = config or {}
        self.original_config = config.copy()
        
        # Rufe Basisklassenkonstruktor auf
        super().__init__(config)
        
        # Ersetze LSTM durch GRU
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # GRU-spezifische Konfiguration
        self.attention_enabled = self.config.get("attention_enabled", False)
        
        # Wenn Attention aktiviert ist, füge Attention-Layer hinzu
        if self.attention_enabled:
            attention_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
            self.attention = nn.Linear(attention_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das GRU-Modell
        
        Args:
            x: Eingabetensor [batch_size, seq_len, input_size]
            
        Returns:
            Ausgabetensor [batch_size, num_classes]
        """
        # GRU-Layer
        output, hidden = self.rnn(x)
        
        if self.attention_enabled:
            # Attention-Mechanismus
            attention_weights = F.softmax(self.attention(output).squeeze(-1), dim=1)
            context_vector = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)
            return self.fc(context_vector)
        else:
            # Standard-Implementierung: Verwende letzten Hidden-State
            if self.bidirectional:
                # Konkateniere vorwärts und rückwärts
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            else:
                hidden = hidden[-1, :, :]
            
            return self.fc(hidden)


class AttentionLayer(nn.Module):
    """Attention-Layer für Sequence-to-Sequence-Modelle"""
    
    def __init__(self, hidden_size: int):
        """
        Initialisiert die Attention-Layer
        
        Args:
            hidden_size: Größe des Hidden-States
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch die Attention-Layer
        
        Args:
            hidden: Hidden-State des Decoders [batch_size, hidden_size]
            encoder_outputs: Ausgaben des Encoders [batch_size, seq_len, hidden_size]
            
        Returns:
            Attention-Gewichte [batch_size, seq_len]
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Wiederhole hidden für jedes Wort in der Sequenz
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Konkateniere hidden und encoder_outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Berechne Attention-Scores
        attention = self.v(energy).squeeze(2)
        
        # Normalisiere mit Softmax
        return F.softmax(attention, dim=1)


class Seq2SeqModel(BaseModel):
    """Sequence-to-Sequence-Modell mit Attention für Q-LOGIK"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert das Seq2Seq-Modell
        
        Args:
            config: Konfigurationsobjekt für das Modell
        """
        super().__init__(config)
        
        # Konfigurationsparameter
        self.input_size = self.config.get("input_size", 300)
        self.hidden_size = self.config.get("hidden_size", 256)
        self.output_size = self.config.get("output_size", 300)
        self.num_layers = self.config.get("num_layers", 2)
        self.dropout = self.config.get("dropout", 0.5)
        self.bidirectional = self.config.get("bidirectional", True)
        self.rnn_type = self.config.get("rnn_type", "lstm").lower()
        self.attention_enabled = self.config.get("attention_enabled", True)
        
        # Encoder
        if self.rnn_type == "gru":
            self.encoder = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.encoder = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            )
        
        # Decoder
        decoder_input_size = self.output_size
        decoder_hidden_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        if self.rnn_type == "gru":
            self.decoder = nn.GRU(
                input_size=decoder_input_size,
                hidden_size=decoder_hidden_size,
                num_layers=1,
                batch_first=True
            )
        else:
            self.decoder = nn.LSTM(
                input_size=decoder_input_size,
                hidden_size=decoder_hidden_size,
                num_layers=1,
                batch_first=True
            )
        
        # Attention
        if self.attention_enabled:
            self.attention = AttentionLayer(decoder_hidden_size)
        
        # Ausgabeschicht
        self.fc_out = nn.Linear(decoder_hidden_size, self.output_size)
    
    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward-Pass durch das Seq2Seq-Modell
        
        Args:
            src: Quellsequenz [batch_size, src_len, input_size]
            trg: Zielsequenz [batch_size, trg_len, output_size]
            teacher_forcing_ratio: Wahrscheinlichkeit für Teacher Forcing
            
        Returns:
            Ausgabesequenz [batch_size, trg_len, output_size]
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        
        # Tensor für Ausgaben initialisieren
        outputs = torch.zeros(batch_size, trg_len, self.output_size).to(self.device)
        
        # Encoder
        if self.rnn_type == "gru":
            encoder_outputs, hidden = self.encoder(src)
            
            # Wenn bidirektional, kombiniere letzte Hidden-States
            if self.bidirectional:
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)
        else:
            encoder_outputs, (hidden, cell) = self.encoder(src)
            
            # Wenn bidirektional, kombiniere letzte Hidden-States
            if self.bidirectional:
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).unsqueeze(0)
                cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1).unsqueeze(0)
        
        # Erste Eingabe für Decoder ist der erste Token der Zielsequenz
        decoder_input = trg[:, 0, :].unsqueeze(1)
        
        # Decoder
        for t in range(1, trg_len):
            # Decoder-Schritt
            if self.rnn_type == "gru":
                decoder_output, hidden = self.decoder(decoder_input, hidden)
            else:
                decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            
            # Attention
            if self.attention_enabled:
                attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                decoder_output = torch.cat((decoder_output, context), dim=2)
            
            # Ausgabe
            prediction = self.fc_out(decoder_output.squeeze(1))
            outputs[:, t, :] = prediction
            
            # Teacher Forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = trg[:, t, :].unsqueeze(1) if teacher_force else prediction.unsqueeze(1)
        
        return outputs


class TransformerEncoderLayer(nn.Module):
    """Transformer-Encoder-Layer für Q-LOGIK"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Initialisiert einen Transformer-Encoder-Layer
        
        Args:
            d_model: Dimensionalität des Modells
            nhead: Anzahl der Attention-Heads
            dim_feedforward: Dimensionalität des Feedforward-Netzwerks
            dropout: Dropout-Rate
        """
        super().__init__()
        
        # Multi-Head Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward-Netzwerk
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalisierung
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward-Pass durch den Transformer-Encoder-Layer
        
        Args:
            src: Eingabetensor [batch_size, seq_len, d_model]
            src_mask: Maske für Padding [batch_size, seq_len, seq_len]
            
        Returns:
            Ausgabetensor [batch_size, seq_len, d_model]
        """
        # Self-Attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer-Decoder-Layer für Q-LOGIK"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Initialisiert einen Transformer-Decoder-Layer
        
        Args:
            d_model: Dimensionalität des Modells
            nhead: Anzahl der Attention-Heads
            dim_feedforward: Dimensionalität des Feedforward-Netzwerks
            dropout: Dropout-Rate
        """
        super().__init__()
        
        # Multi-Head Attention (self-attention)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Multi-Head Attention (encoder-decoder attention)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward-Netzwerk
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalisierung
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward-Pass durch den Transformer-Decoder-Layer
        
        Args:
            tgt: Zieltensor [batch_size, tgt_len, d_model]
            memory: Speichertensor vom Encoder [batch_size, src_len, d_model]
            tgt_mask: Maske für Zielsequenz [batch_size, tgt_len, tgt_len]
            memory_mask: Maske für Speicher [batch_size, tgt_len, src_len]
            
        Returns:
            Ausgabetensor [batch_size, tgt_len, d_model]
        """
        # Self-Attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Encoder-Decoder-Attention
        tgt2, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerModel(BaseModel):
    """Transformer-Modell für Q-LOGIK"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert das Transformer-Modell
        
        Args:
            config: Konfigurationsobjekt für das Modell
        """
        super().__init__(config)
        
        # Konfigurationsparameter
        self.input_size = self.config.get("input_size", 512)
        self.d_model = self.config.get("d_model", 512)
        self.nhead = self.config.get("nhead", 8)
        self.num_encoder_layers = self.config.get("num_encoder_layers", 6)
        self.num_decoder_layers = self.config.get("num_decoder_layers", 6)
        self.dim_feedforward = self.config.get("dim_feedforward", 2048)
        self.dropout = self.config.get("dropout", 0.1)
        self.output_size = self.config.get("output_size", 512)
        
        # Embedding-Layer
        self.embedding = nn.Linear(self.input_size, self.d_model)
        
        # Positional Encoding
        self.pos_encoder = self._create_positional_encoding()
        
        # Encoder-Layer
        encoder_layers = []
        for _ in range(self.num_encoder_layers):
            encoder_layers.append(
                TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout
                )
            )
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # Decoder-Layer
        decoder_layers = []
        for _ in range(self.num_decoder_layers):
            decoder_layers.append(
                TransformerDecoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout
                )
            )
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # Ausgabeschicht
        self.fc_out = nn.Linear(self.d_model, self.output_size)
    
    def _create_positional_encoding(self) -> nn.Module:
        """
        Erstellt Positional Encoding für den Transformer
        
        Returns:
            Positional Encoding Modul
        """
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                
                position = torch.arange(max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
                pe = torch.zeros(max_len, d_model)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + self.pe[:, :x.size(1), :]
                return self.dropout(x)
        
        return PositionalEncoding(self.d_model, self.dropout)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generiert eine quadratische Maske für die Decoder-Self-Attention
        
        Args:
            sz: Größe der Maske
            
        Returns:
            Maske [sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das Transformer-Modell
        
        Args:
            src: Quellsequenz [batch_size, src_len, input_size]
            tgt: Zielsequenz [batch_size, tgt_len, input_size]
            
        Returns:
            Ausgabesequenz [batch_size, tgt_len, output_size]
        """
        # Embedding
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        
        # Positional Encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Masken erstellen
        tgt_len = tgt.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len)
        
        # Encoder
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory)
        
        # Decoder
        output = tgt
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, memory, tgt_mask=tgt_mask)
        
        # Ausgabe
        output = self.fc_out(output)
        
        return output


# Modell-Factory für RNN-Modelle
def create_rnn_model(model_type: str, config: Dict[str, Any] = None) -> BaseModel:
    """
    Erstellt ein RNN-Modell
    
    Args:
        model_type: Typ des RNN-Modells (lstm, gru, seq2seq, transformer)
        config: Konfigurationsobjekt für das Modell
        
    Returns:
        RNN-Modellinstanz
    """
    config = config or {}
    
    if model_type.lower() == "lstm":
        return LSTMModel(config)
    elif model_type.lower() == "gru":
        return GRUModel(config)
    elif model_type.lower() == "seq2seq":
        return Seq2SeqModel(config)
    elif model_type.lower() == "transformer":
        return TransformerModel(config)
    else:
        logger.warning(f"Unbekannter RNN-Modelltyp: {model_type}, verwende LSTM")
        return LSTMModel(config)


if __name__ == "__main__":
    # Beispiel für die Verwendung der RNN-Modelle
    logging.basicConfig(level=logging.INFO)
    
    # Testdaten
    batch_size = 4
    seq_len = 20
    input_size = 300
    
    # LSTM-Modell
    lstm_config = {
        "input_size": input_size,
        "hidden_size": 256,
        "num_classes": 5,
        "attention_enabled": True
    }
    lstm_model = create_rnn_model("lstm", lstm_config)
    lstm_input = torch.randn(batch_size, seq_len, input_size)
    lstm_output = lstm_model(lstm_input)
    print(f"LSTM Ausgabeform: {lstm_output.shape}")
    
    # GRU-Modell
    gru_config = {
        "input_size": input_size,
        "hidden_size": 256,
        "num_classes": 5
    }
    gru_model = create_rnn_model("gru", gru_config)
    gru_input = torch.randn(batch_size, seq_len, input_size)
    gru_output = gru_model(gru_input)
    print(f"GRU Ausgabeform: {gru_output.shape}")
    
    # Seq2Seq-Modell
    seq2seq_config = {
        "input_size": input_size,
        "output_size": input_size,
        "hidden_size": 256
    }
    seq2seq_model = create_rnn_model("seq2seq", seq2seq_config)
    src_seq = torch.randn(batch_size, seq_len, input_size)
    tgt_seq = torch.randn(batch_size, seq_len, input_size)
    seq2seq_output = seq2seq_model(src_seq, tgt_seq, teacher_forcing_ratio=0.5)
    print(f"Seq2Seq Ausgabeform: {seq2seq_output.shape}")
    
    # Transformer-Modell
    transformer_config = {
        "input_size": input_size,
        "output_size": input_size,
        "d_model": 512,
        "nhead": 8
    }
    transformer_model = create_rnn_model("transformer", transformer_config)
    src_seq = torch.randn(batch_size, seq_len, input_size)
    tgt_seq = torch.randn(batch_size, seq_len, input_size)
    transformer_output = transformer_model(src_seq, tgt_seq)
    print(f"Transformer Ausgabeform: {transformer_output.shape}")
