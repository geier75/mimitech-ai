#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - Modellklassen

Diese Datei implementiert die optimierten Modellklassen für die T-Mathematics Engine,
einschließlich Attention-Mechanismen und Mixture-of-Experts-Implementierungen.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


class OptimizedMultiHeadAttention(nn.Module):
    """
    Optimierte Multi-Head-Attention-Implementierung für die T-Mathematics Engine.
    
    Diese Klasse bietet eine für AMD RDNA3 und Apple Silicon optimierte
    Implementierung des Multi-Head-Attention-Mechanismus.
    """
    
    def __init__(self,
                embed_dim: int,
                num_heads: int,
                dropout: float = 0.1,
                bias: bool = True,
                engine = None):
        """
        Initialisiert die Multi-Head-Attention-Schicht.
        
        Args:
            embed_dim: Einbettungsdimension
            num_heads: Anzahl der Attention-Köpfe
            dropout: Dropout-Wahrscheinlichkeit
            bias: Ob Bias verwendet werden soll
            engine: Referenz zur T-Mathematics Engine
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim muss durch num_heads teilbar sein"
        
        # Engine-Referenz
        self.engine = engine
        
        # Projektionsmatrizen
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Initialisierung
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialisiert die Parameter der Schicht"""
        # Xavier-Initialisierung für bessere Konvergenz
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, 
               query: torch.Tensor,
               key: Optional[torch.Tensor] = None,
               value: Optional[torch.Tensor] = None,
               attn_mask: Optional[torch.Tensor] = None,
               key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Führt die Multi-Head-Attention-Berechnung durch.
        
        Args:
            query: Query-Tensor [batch_size, seq_len, embed_dim]
            key: Key-Tensor [batch_size, seq_len, embed_dim]
            value: Value-Tensor [batch_size, seq_len, embed_dim]
            attn_mask: Aufmerksamkeitsmaske [seq_len, seq_len]
            key_padding_mask: Padding-Maske [batch_size, seq_len]
            
        Returns:
            Ausgabe der Attention-Berechnung [batch_size, seq_len, embed_dim]
        """
        # Verwende query als key und value, falls nicht angegeben (Self-Attention)
        if key is None:
            key = query
        if value is None:
            value = key
            
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)
        
        # Projektionen
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape für Multi-Head-Attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Kombiniere Masken
        mask = None
        if attn_mask is not None or key_padding_mask is not None:
            mask = torch.ones(batch_size, 1, tgt_len, src_len, device=query.device)
            
            if attn_mask is not None:
                mask = mask * attn_mask.unsqueeze(0).unsqueeze(0)
                
            if key_padding_mask is not None:
                mask = mask * key_padding_mask.unsqueeze(1).unsqueeze(2)
        
        # Berechne Attention mit der Engine
        if self.engine is not None:
            attn_output, _ = self.engine.attention(q, k, v, mask=mask, dropout_p=self.dropout)
        else:
            # Fallback für den Fall, dass keine Engine verfügbar ist
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
                
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape zurück
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        
        # Ausgabeprojektion
        output = self.out_proj(attn_output)
        
        return output


class OptimizedFeedForward(nn.Module):
    """
    Optimierte Feed-Forward-Implementierung für die T-Mathematics Engine.
    
    Diese Klasse bietet eine für AMD RDNA3 und Apple Silicon optimierte
    Implementierung des Feed-Forward-Netzwerks.
    """
    
    def __init__(self,
                input_dim: int,
                hidden_dim: int,
                output_dim: int,
                activation: str = "gelu",
                dropout: float = 0.1,
                engine = None):
        """
        Initialisiert die Feed-Forward-Schicht.
        
        Args:
            input_dim: Eingabedimension
            hidden_dim: Versteckte Dimension
            output_dim: Ausgabedimension
            activation: Aktivierungsfunktion
            dropout: Dropout-Wahrscheinlichkeit
            engine: Referenz zur T-Mathematics Engine
        """
        super().__init__()
        
        # Robuste Parametervalidierung
        if input_dim is None or input_dim <= 0:
            raise ValueError(f"input_dim muss positiv sein, erhalten: {input_dim}")
        if hidden_dim is None or hidden_dim <= 0:
            raise ValueError(f"hidden_dim muss positiv sein, erhalten: {hidden_dim}")
        if output_dim is None or output_dim <= 0:
            raise ValueError(f"output_dim muss positiv sein, erhalten: {output_dim}")
        
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.activation_name = activation
        self.dropout = dropout
        
        # Engine-Referenz
        self.engine = engine
        
        # Schichten mit expliziter Typkonvertierung
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Aktivierungsfunktion
        if engine is not None:
            self.activation = engine.compat_engine.get_activation_fn(activation)
        else:
            if activation.lower() == "gelu":
                self.activation = F.gelu
            elif activation.lower() == "relu":
                self.activation = F.relu
            elif activation.lower() == "silu" or activation.lower() == "swish":
                self.activation = F.silu
            else:
                raise ValueError(f"Unbekannte Aktivierungsfunktion: {activation}")
        
        # Initialisierung
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialisiert die Parameter der Schicht"""
        # Kaiming-Initialisierung für ReLU-ähnliche Aktivierungen
        if self.activation_name.lower() in ["relu", "gelu", "silu", "swish"]:
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Führt die Feed-Forward-Berechnung durch.
        
        Args:
            x: Eingabetensor [batch_size, seq_len, input_dim]
            
        Returns:
            Ausgabe der Feed-Forward-Berechnung [batch_size, seq_len, output_dim]
        """
        # Erste Schicht
        h = self.fc1(x)
        
        # Aktivierungsfunktion
        h = self.activation(h)
        
        # Dropout
        h = self.dropout_layer(h)
        
        # Zweite Schicht
        output = self.fc2(h)
        
        return output


class Expert(nn.Module):
    """
    Experte für die Mixture-of-Experts-Architektur.
    
    Diese Klasse implementiert einen einzelnen Experten, der aus einem
    Feed-Forward-Netzwerk besteht.
    """
    
    def __init__(self,
                input_dim: int,
                hidden_dim: int,
                output_dim: int,
                activation: str = "gelu",
                engine = None):
        """
        Initialisiert den Experten.
        
        Args:
            input_dim: Eingabedimension
            hidden_dim: Versteckte Dimension
            output_dim: Ausgabedimension
            activation: Aktivierungsfunktion
            engine: Referenz zur T-Mathematics Engine
        """
        super().__init__()
        
        # Erstelle das Feed-Forward-Netzwerk
        # Verwende immer OptimizedFeedForward (Engine-Methode existiert nicht)
        self.ffn = OptimizedFeedForward(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=activation,
            dropout=0.0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Führt die Berechnung des Experten durch.
        
        Args:
            x: Eingabetensor [batch_size, seq_len, input_dim]
            
        Returns:
            Ausgabe des Experten [batch_size, seq_len, output_dim]
        """
        return self.ffn(x)


class MixtureOfExperts(nn.Module):
    """
    Mixture-of-Experts-Implementierung für die T-Mathematics Engine.
    
    Diese Klasse implementiert eine Mixture-of-Experts-Schicht, die mehrere
    Experten kombiniert, um eine Ausgabe zu erzeugen.
    """
    
    def __init__(self,
                input_dim: int,
                output_dim: int,
                num_experts: int,
                hidden_dim: int = None,
                expert_capacity: int = None,
                router_jitter_noise: float = 0.01,
                activation: str = "gelu",
                engine = None,
                k: int = 2,  # Backward compatibility
                top_k: int = None):
        """
        Initialisiert die MoE-Schicht.
        
        Args:
            input_dim: Eingabedimension
            hidden_dim: Versteckte Dimension
            output_dim: Ausgabedimension
            num_experts: Anzahl der Experten
            expert_capacity: Kapazität jedes Experten
            router_jitter_noise: Rauschfaktor für den Router
            activation: Aktivierungsfunktion
            engine: Referenz zur T-Mathematics Engine
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else (input_dim * 4)  # Default hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity if expert_capacity is not None else max(1, input_dim // num_experts)  # Default expert_capacity
        self.router_jitter_noise = router_jitter_noise
        self.top_k = top_k if top_k is not None else k  # Use top_k if provided, otherwise use k
        self.last_routing_probs = None  # Store last routing probabilities
        
        # Validiere Parameter
        if self.hidden_dim <= 0:
            self.hidden_dim = input_dim * 4
        if self.expert_capacity <= 0:
            self.expert_capacity = max(1, input_dim // num_experts)
        
        # Engine-Referenz
        self.engine = engine
        
        # Router-Gewichte
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Experten
        self.experts = nn.ModuleList([
            Expert(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,  # Verwende validierte hidden_dim
                output_dim=output_dim,
                activation=activation,
                engine=engine
            )
            for _ in range(num_experts)
        ])
        
        # Initialisierung
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialisiert die Parameter der Schicht"""
        # Initialisiere Router-Gewichte
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Führt die MoE-Berechnung durch.
        
        Args:
            x: Eingabetensor [batch_size, seq_len, input_dim]
            
        Returns:
            Ausgabe der MoE-Berechnung [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Reshape für die Routing-Berechnung
        x_flat = x.view(-1, self.input_dim)  # [batch_size * seq_len, input_dim]
        
        # Berechne Routing-Entscheidung
        if self.engine is not None:
            router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
            routing_weights, routing_indices = self.engine.route_to_experts(
                x_flat, 
                self.router.weight.t(),
                top_k=min(self.top_k, self.num_experts),
                noise_epsilon=self.router_jitter_noise if self.training else 0.0
            )
        else:
            # Fallback für den Fall, dass keine Engine verfügbar ist
            router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
            
            # Füge Rauschen für besseres Load-Balancing hinzu
            if self.training and self.router_jitter_noise > 0:
                router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_noise
                
            # Berechne Routing-Wahrscheinlichkeiten
            routing_weights = F.softmax(router_logits, dim=-1)
            
            # Wähle die Top-K Experten aus
            actual_top_k = min(self.top_k, self.num_experts)
            routing_weights, routing_indices = torch.topk(routing_weights, actual_top_k, dim=-1)
            
            # Normalisiere die Top-K Wahrscheinlichkeiten
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Speichere Routing-Wahrscheinlichkeiten für Analyse
        self.last_routing_probs = routing_weights.detach().clone()
        
        # Reshape zurück
        routing_weights = routing_weights.view(batch_size, seq_len, -1)  # [batch_size, seq_len, top_k]
        routing_indices = routing_indices.view(batch_size, seq_len, -1)  # [batch_size, seq_len, top_k]
        
        # Berechne die Ausgaben der Experten
        expert_outputs = []
        for expert_idx in range(self.num_experts):
            # Finde Token, die diesen Experten verwenden
            expert_mask = (routing_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]
            
            if expert_mask.any():
                # Extrahiere relevante Token
                expert_input = x[expert_mask]  # [num_tokens, input_dim]
                
                # Berechne Expertenausgabe
                expert_output = self.experts[expert_idx](expert_input)  # [num_tokens, output_dim]
                
                # Erstelle einen leeren Ausgabetensor
                full_expert_output = torch.zeros(
                    batch_size, seq_len, self.output_dim,
                    device=x.device, dtype=x.dtype
                )
                
                # Fülle den Ausgabetensor mit den berechneten Werten
                full_expert_output[expert_mask] = expert_output
                
                expert_outputs.append(full_expert_output)
            else:
                # Wenn kein Token diesen Experten verwendet, erstelle einen leeren Tensor
                expert_outputs.append(torch.zeros(
                    batch_size, seq_len, self.output_dim,
                    device=x.device, dtype=x.dtype
                ))
        
        # Kombiniere die Expertenausgaben
        if self.engine is not None:
            output = self.engine.combine_expert_outputs(
                expert_outputs=expert_outputs,
                expert_weights=routing_weights,
                normalize_weights=True
            )
        else:
            # Fallback für den Fall, dass keine Engine verfügbar ist
            # Stapel die Expertenausgaben
            stacked_outputs = torch.stack(expert_outputs, dim=-2)  # [batch_size, seq_len, num_experts, output_dim]
            
            # Erweitere die Gewichte für das Broadcasting
            weights = routing_weights.unsqueeze(-1)  # [batch_size, seq_len, top_k, 1]
            
            # Erstelle einen Tensor mit den richtigen Indizes
            indices = routing_indices.unsqueeze(-1).expand(-1, -1, -1, self.output_dim)  # [batch_size, seq_len, top_k, output_dim]
            
            # Initialisiere den Ausgabetensor mit Nullen
            output = torch.zeros(
                batch_size, seq_len, self.output_dim,
                device=x.device, dtype=x.dtype
            )
            
            # Fülle den Ausgabetensor mit den gewichteten Expertenausgaben
            for i in range(routing_indices.size(-1)):
                # Hole die Expertenindizes für diese Position
                expert_idx = routing_indices[:, :, i]  # [batch_size, seq_len]
                weight = weights[:, :, i, :]  # [batch_size, seq_len, 1]
                
                # Iteriere über alle Experten und addiere deren gewichtete Ausgaben
                for batch_idx in range(batch_size):
                    for seq_idx in range(seq_len):
                        exp_idx = expert_idx[batch_idx, seq_idx].item()
                        if 0 <= exp_idx < len(expert_outputs):
                            output[batch_idx, seq_idx] += (
                                expert_outputs[exp_idx][batch_idx, seq_idx] * 
                                weight[batch_idx, seq_idx, 0]
                            )
        
        return output


class TMathTransformerLayer(nn.Module):
    """
    Optimierte Transformer-Schicht für die T-Mathematics Engine.
    
    Diese Klasse implementiert eine für AMD RDNA3 und Apple Silicon optimierte
    Transformer-Schicht, die Multi-Head-Attention und Feed-Forward-Netzwerke
    kombiniert.
    """
    
    def __init__(self,
                d_model: int,
                nhead: int,
                dim_feedforward: int,
                dropout: float = 0.1,
                activation: str = "gelu",
                layer_norm_eps: float = 1e-5,
                use_moe: bool = False,
                num_experts: int = 8,
                expert_capacity: int = 64,
                engine = None):
        """
        Initialisiert die Transformer-Schicht.
        
        Args:
            d_model: Modellgröße
            nhead: Anzahl der Attention-Köpfe
            dim_feedforward: Dimension des Feed-Forward-Netzwerks
            dropout: Dropout-Wahrscheinlichkeit
            activation: Aktivierungsfunktion
            layer_norm_eps: Epsilon für Layer-Normalisierung
            use_moe: Ob MoE verwendet werden soll
            num_experts: Anzahl der Experten (nur bei use_moe=True)
            expert_capacity: Kapazität jedes Experten (nur bei use_moe=True)
            engine: Referenz zur T-Mathematics Engine
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.use_moe = use_moe
        
        # Engine-Referenz
        self.engine = engine
        
        # Self-Attention
        if engine is not None:
            self.self_attn = engine.create_attention_layer(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                bias=True
            )
        else:
            self.self_attn = OptimizedMultiHeadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                bias=True
            )
        
        # Feed-Forward oder MoE
        if use_moe:
            if engine is not None:
                self.feed_forward = engine.create_moe_layer(
                    input_dim=d_model,
                    hidden_dim=dim_feedforward,
                    output_dim=d_model,
                    num_experts=num_experts,
                    expert_capacity=expert_capacity,
                    activation=activation
                )
            else:
                self.feed_forward = MixtureOfExperts(
                    input_dim=d_model,
                    hidden_dim=dim_feedforward,
                    output_dim=d_model,
                    num_experts=num_experts,
                    expert_capacity=expert_capacity,
                    activation=activation
                )
        else:
            if engine is not None:
                self.feed_forward = engine.create_feedforward_layer(
                    input_dim=d_model,
                    hidden_dim=dim_feedforward,
                    output_dim=d_model,
                    activation=activation,
                    dropout=dropout
                )
            else:
                self.feed_forward = OptimizedFeedForward(
                    input_dim=d_model,
                    hidden_dim=dim_feedforward,
                    output_dim=d_model,
                    activation=activation,
                    dropout=dropout
                )
        
        # Layer-Normalisierung
        if engine is not None:
            self.norm1 = engine.compat_engine.create_layer_norm(d_model, eps=layer_norm_eps)
            self.norm2 = engine.compat_engine.create_layer_norm(d_model, eps=layer_norm_eps)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
               src: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Führt die Transformer-Schicht-Berechnung durch.
        
        Args:
            src: Eingabetensor [batch_size, seq_len, d_model]
            src_mask: Aufmerksamkeitsmaske [seq_len, seq_len]
            src_key_padding_mask: Padding-Maske [batch_size, seq_len]
            
        Returns:
            Ausgabe der Transformer-Schicht [batch_size, seq_len, d_model]
        """
        # Self-Attention
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        
        # Feed-Forward oder MoE
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout(src2)
        
        return src
