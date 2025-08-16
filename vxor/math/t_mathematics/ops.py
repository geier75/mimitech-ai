#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-Mathematics Engine - Mathematische Operationen

Diese Datei implementiert die grundlegenden mathematischen Operationen,
die von der T-Mathematics Engine verwendet werden. Optimiert für
Apple Silicon M4 Max und AMD RDNA3 Hardware.

Copyright (c) 2025 MISO Team. Alle Rechte vorbehalten.
"""

import torch
import numpy as np
import math
from typing import List, Tuple, Optional, Union, Dict, Any

# Prüfe auf Apple Silicon
is_apple_silicon = False
try:
    import platform
    is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
except:
    pass

# Prüfe auf AMD-Optimierungen
has_amd_optimizations = False
try:
    if hasattr(torch, 'version') and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        has_amd_optimizations = True
except:
    pass


def tensor_svd(tensor: torch.Tensor, k: Optional[int] = None, 
              implementation: str = "randomized") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Führt eine SVD-Zerlegung eines Tensors durch und gibt die wichtigsten Komponenten zurück.
    Optimiert für verschiedene Hardware-Plattformen.
    
    Args:
        tensor: Eingabetensor für die SVD
        k: Anzahl der zurückzugebenden Komponenten, oder None für alle
        implementation: Implementierungsmethode ("full", "randomized", "truncated")
        
    Returns:
        Tuple aus (U, S, V) Matrizen
    """
    # Stellen Sie sicher, dass der Tensor mindestens 2D ist
    if len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    
    # Bestimme die maximale Anzahl von Komponenten
    max_k = min(tensor.shape)
    if k is not None:
        k = min(k, max_k)
    
    # Wähle die Implementierung basierend auf Hardware und Größe
    if implementation == "randomized" and (tensor.numel() > 10000 or k is not None):
        # Randomisierte SVD für große Tensoren oder wenn nur k Komponenten benötigt werden
        return _randomized_svd(tensor, k)
    elif implementation == "truncated" and k is not None:
        # Truncated SVD wenn nur k Komponenten benötigt werden
        return _truncated_svd(tensor, k)
    else:
        # Volle SVD für kleine Tensoren oder wenn alle Komponenten benötigt werden
        try:
            U, S, V = torch.svd(tensor, some=k is not None)
            if k is not None:
                U = U[:, :k]
                S = S[:k]
                V = V[:, :k]
            return U, S, V
        except Exception as e:
            # Fallback auf NumPy SVD bei Fehler
            # Konvertiere zu float32 falls float16 (NumPy unterstützt float16 nicht für SVD)
            original_dtype = tensor.dtype
            if tensor.dtype == torch.float16:
                tensor_for_svd = tensor.float()
            else:
                tensor_for_svd = tensor
            
            tensor_cpu = tensor_for_svd.cpu().numpy()
            U, S, Vh = np.linalg.svd(tensor_cpu, full_matrices=False)
            if k is not None:
                U = U[:, :k]
                S = S[:k]
                Vh = Vh[:k, :]
            
            # Konvertiere zurück zum ursprünglichen dtype
            U_tensor = torch.from_numpy(U).to(device=tensor.device, dtype=original_dtype)
            S_tensor = torch.from_numpy(S).to(device=tensor.device, dtype=original_dtype)
            V_tensor = torch.from_numpy(Vh.T).to(device=tensor.device, dtype=original_dtype)
            return U_tensor, S_tensor, V_tensor


def _randomized_svd(tensor: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implementiert eine randomisierte SVD für große Tensoren.
    
    Args:
        tensor: Eingabetensor
        k: Anzahl der Komponenten
        
    Returns:
        Tuple aus (U, S, V) Matrizen
    """
    if k is None:
        k = min(tensor.shape)
    
    m, n = tensor.shape
    transpose = False
    
    # Für bessere Effizienz, transponiere wenn m > n
    if m > n:
        tensor = tensor.T
        m, n = tensor.shape
        transpose = True
    
    # Überabtastungsfaktor für numerische Stabilität
    oversampling = min(10, n - k)
    n_components = min(k + oversampling, n)
    
    # Schritt 1: Erzeuge eine zufällige Matrix
    random_matrix = torch.randn(n, n_components, device=tensor.device)
    
    # Schritt 2: Berechne Y = A * Omega
    Y = tensor @ random_matrix
    
    # Schritt 3: QR-Zerlegung von Y
    Q, _ = torch.qr(Y)
    
    # Schritt 4: Berechne B = Q^T * A
    B = Q.T @ tensor
    
    # Schritt 5: SVD von B
    Uhat, S, V = torch.svd(B)
    
    # Schritt 6: Berechne U = Q * Uhat
    U = Q @ Uhat
    
    # Beschneide auf k Komponenten
    U = U[:, :k]
    S = S[:k]
    V = V[:, :k]
    
    # Wenn ursprünglich transponiert, tausche U und V
    if transpose:
        return V, S, U
    else:
        return U, S, V


def _truncated_svd(tensor: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implementiert eine truncated SVD für Tensoren.
    
    Args:
        tensor: Eingabetensor
        k: Anzahl der Komponenten
        
    Returns:
        Tuple aus (U, S, V) Matrizen
    """
    m, n = tensor.shape
    transpose = False
    
    # Für bessere Effizienz, transponiere wenn m > n
    if m > n:
        tensor = tensor.T
        m, n = tensor.shape
        transpose = True
    
    # Berechne A^T * A für Eigenwertzerlegung
    if n > 1000:  # Für große Matrizen, verwende eine effizientere Methode
        AtA = tensor.T @ tensor
        eigenvalues, eigenvectors = torch.symeig(AtA, eigenvectors=True)
        
        # Sortiere Eigenwerte in absteigender Reihenfolge
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Berechne Singulärwerte und rechte Singulärvektoren
        S = torch.sqrt(eigenvalues[:k])
        V = eigenvectors[:, :k]
        
        # Berechne linke Singulärvektoren
        U = tensor @ V / S.unsqueeze(0)
    else:
        # Für kleinere Matrizen, verwende direkt torch.svd
        U, S, V = torch.svd(tensor, some=True)
        U = U[:, :k]
        S = S[:k]
        V = V[:, :k]
    
    # Wenn ursprünglich transponiert, tausche U und V
    if transpose:
        return V, S, U
    else:
        return U, S, V


def amd_optimized_matmul(a: torch.Tensor, b: torch.Tensor, 
                        optimize_for: str = "auto") -> torch.Tensor:
    """
    Hardware-optimierte Matrixmultiplikation.
    
    Args:
        a: Erster Tensor
        b: Zweiter Tensor
        optimize_for: Hardware-Optimierung ("amd", "apple", "auto")
        
    Returns:
        Ergebnis der Matrixmultiplikation
    """
    # Automatische Hardware-Erkennung
    if optimize_for == "auto":
        if has_amd_optimizations:
            optimize_for = "amd"
        elif is_apple_silicon:
            optimize_for = "apple"
        else:
            optimize_for = "default"
    
    # AMD-spezifische Optimierungen
    if optimize_for == "amd" and has_amd_optimizations:
        # Optimierungen für AMD-Hardware
        
        # Stellen Sie sicher, dass die Tensoren contiguous sind
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
        
        # Für AMD-GPUs, verwende spezifische Speicherlayouts
        if a.dim() >= 4 and b.dim() >= 4:
            # Für 4D+ Tensoren, verwende channels_last Format für bessere Performance
            a = a.contiguous(memory_format=torch.channels_last)
            b = b.contiguous(memory_format=torch.channels_last)
            
        # Verwende Mixed Precision für bessere Performance
        if a.dtype == torch.float32 and b.dtype == torch.float32:
            with torch.cuda.amp.autocast(enabled=True):
                return torch.matmul(a, b)
        
        return torch.matmul(a, b)
    
    # Apple Silicon-spezifische Optimierungen
    elif optimize_for == "apple" and is_apple_silicon:
        # Optimierungen für Apple Neural Engine
        
        # Stellen Sie sicher, dass die Tensoren auf dem richtigen Gerät sind
        if a.device.type != "mps" and torch.backends.mps.is_available():
            a = a.to("mps")
        if b.device.type != "mps" and torch.backends.mps.is_available():
            b = b.to("mps")
        
        # Führe die Matrixmultiplikation durch
        result = torch.matmul(a, b)
        
        return result
    
    # Standard-Implementierung
    else:
        return torch.matmul(a, b)


def moe_routing(inputs: torch.Tensor, 
               router_weights: torch.Tensor, 
               top_k: int = 2,
               noise_epsilon: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Berechnet die Routing-Entscheidung für die MoE-Architektur.
    
    Args:
        inputs: Einbettungen der Eingabe [batch_size, seq_len, hidden_dim]
        router_weights: Routing-Gewichte [hidden_dim, num_experts]
        top_k: Anzahl der Experten pro Token
        noise_epsilon: Rauschfaktor für Load-Balancing
        
    Returns:
        Tuple aus (routing_probabilities, expert_indices)
    """
    # Berechne Routing-Logits
    routing_logits = torch.matmul(inputs, router_weights)  # [batch_size, seq_len, num_experts]
    
    # Füge Rauschen für besseres Load-Balancing hinzu
    if noise_epsilon > 0 and inputs.requires_grad:
        routing_noise = torch.randn_like(routing_logits) * noise_epsilon
        routing_logits = routing_logits + routing_noise
    
    # Berechne Routing-Wahrscheinlichkeiten
    routing_probs = torch.softmax(routing_logits, dim=-1)
    
    # Wähle die Top-K Experten aus
    top_k_probs, top_k_indices = torch.topk(routing_probs, top_k, dim=-1)
    
    # Normalisiere die Top-K Wahrscheinlichkeiten
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    
    return top_k_probs, top_k_indices


def mix_experts_outputs(expert_outputs: List[torch.Tensor], 
                      expert_weights: torch.Tensor,
                      normalize_weights: bool = True) -> torch.Tensor:
    """
    Mischt die Ausgaben mehrerer Experten basierend auf den Gewichten.
    
    Args:
        expert_outputs: Liste der Expertenausgaben, jeder mit Form [batch_size, seq_len, hidden_dim]
        expert_weights: Gewichte für jeden Experten [batch_size, seq_len, num_experts]
        normalize_weights: Ob die Gewichte normalisiert werden sollen
        
    Returns:
        Gewichtete Summe der Expertenausgaben
    """
    # Stapel die Expertenausgaben
    stacked_outputs = torch.stack(expert_outputs, dim=-2)  # [batch_size, seq_len, num_experts, hidden_dim]
    
    # Normalisiere die Gewichte, falls erforderlich
    if normalize_weights:
        weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
    else:
        weights = expert_weights
    
    # Erweitere die Gewichte für das Broadcasting
    weights = weights.unsqueeze(-1)  # [batch_size, seq_len, num_experts, 1]
    
    # Gewichtete Summe
    weighted_outputs = stacked_outputs * weights  # [batch_size, seq_len, num_experts, hidden_dim]
    mixed_outputs = weighted_outputs.sum(dim=-2)  # [batch_size, seq_len, hidden_dim]
    
    return mixed_outputs


def positional_encoding(seq_len: int, dim: int, device: torch.device,
                       max_len: int = 10000) -> torch.Tensor:
    """
    Erzeugt Positionscodierungen für Transformers.
    
    Args:
        seq_len: Länge der Sequenz
        dim: Dimension der Codierung
        device: Gerät, auf dem der Tensor erstellt werden soll
        max_len: Maximale Sequenzlänge für die Skalierung
        
    Returns:
        Positionscodierungen mit Form [seq_len, dim]
    """
    # Positionen
    positions = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    
    # Dimensionen
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * 
                       -(math.log(max_len) / dim))
    
    # Berechne Positionscodierungen
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    
    return pe


def attention_wrapper(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     mask: Optional[torch.Tensor] = None,
                     dropout_p: float = 0.0,
                     use_flash_attention: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper für verschiedene Attention-Implementierungen.
    
    Args:
        query: Query-Tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key-Tensor [batch_size, num_heads, seq_len, head_dim]
        value: Value-Tensor [batch_size, num_heads, seq_len, head_dim]
        mask: Optionale Aufmerksamkeitsmaske [batch_size, 1, seq_len, seq_len]
        dropout_p: Dropout-Wahrscheinlichkeit
        use_flash_attention: Ob Flash Attention verwendet werden soll
        
    Returns:
        Tuple aus (Attention-Output, Attention-Gewichte)
    """
    # Prüfe, ob Flash Attention verfügbar ist
    has_flash_attention = False
    try:
        import flash_attn
        has_flash_attention = True
    except ImportError:
        pass
    
    # Verwende Flash Attention, wenn verfügbar und gewünscht
    if has_flash_attention and use_flash_attention:
        try:
            from flash_attn import flash_attention_qkvpacked_func
            
            # Bereite QKV für Flash Attention vor
            batch_size, num_heads, seq_len, head_dim = query.shape
            qkv = torch.stack([query, key, value], dim=2)  # [batch_size, num_heads, 3, seq_len, head_dim]
            qkv = qkv.reshape(batch_size, num_heads, 3, seq_len, head_dim)
            
            # Rufe Flash Attention auf
            output, attn_weights = flash_attention_qkvpacked_func(
                qkv, dropout_p=dropout_p, causal=mask is not None
            )
            
            return output, attn_weights
        except Exception as e:
            # Fallback bei Fehler
            print(f"Flash Attention fehlgeschlagen: {e}. Verwende Standard-Attention.")
    
    # Standard-Attention-Implementierung
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
    
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights
