class MLXTensorWrapper:
    """
    Ein Wrapper für MLX-Arrays, der die MISOTensor-Schnittstelle implementiert.
    
    Diese Klasse umhüllt ein MLX-Array und stellt zusätzliche Attribute und Methoden
    bereit, die vom Test-Framework erwartet werden.
    """
    
    def __init__(self, mlx_array):
        """
        Initialisiert den Wrapper mit einem MLX-Array.
        
        Args:
            mlx_array: Das zu umhüllende MLX-Array
        """
        self._data = mlx_array
        self.backend = "mlx"  # Das vom Test erwartete Attribut
        self.shape = mlx_array.shape
    
    def __add__(self, other):
        """Addition"""
        if isinstance(other, MLXTensorWrapper):
            result = self._data + other._data
        else:
            result = self._data + other
        return MLXTensorWrapper(result)
    
    def __matmul__(self, other):
        """Matrix-Multiplikation"""
        import mlx.core as mx
        if isinstance(other, MLXTensorWrapper):
            result = mx.matmul(self._data, other._data)
        else:
            result = mx.matmul(self._data, other)
        return MLXTensorWrapper(result)
    
    def to_numpy(self):
        """Konvertiert zu NumPy-Array"""
        return self._data.__array__()
    
    def exp(self):
        """Exponentielle Funktion"""
        import mlx.core as mx
        return MLXTensorWrapper(mx.exp(self._data))


class TorchTensorWrapper:
    """
    Ein Wrapper für PyTorch-Tensoren, der die MISOTensor-Schnittstelle implementiert.
    
    Diese Klasse umhüllt einen PyTorch-Tensor und stellt zusätzliche Attribute und Methoden
    bereit, die vom Test-Framework erwartet werden.
    """
    
    def __init__(self, torch_tensor):
        """
        Initialisiert den Wrapper mit einem PyTorch-Tensor.
        
        Args:
            torch_tensor: Der zu umhüllende PyTorch-Tensor
        """
        self._data = torch_tensor
        self.backend = "torch"  # Das vom Test erwartete Attribut
        self.shape = torch_tensor.shape
    
    def __add__(self, other):
        """Addition"""
        if isinstance(other, TorchTensorWrapper):
            result = self._data + other._data
        else:
            result = self._data + other
        return TorchTensorWrapper(result)
    
    def __matmul__(self, other):
        """Matrix-Multiplikation"""
        if isinstance(other, TorchTensorWrapper):
            result = self._data @ other._data
        else:
            result = self._data @ other
        return TorchTensorWrapper(result)
    
    def to_numpy(self):
        """Konvertiert zu NumPy-Array"""
        return self._data.detach().cpu().numpy()
    
    def exp(self):
        """Exponentielle Funktion"""
        import torch.nn.functional as F
        return TorchTensorWrapper(torch.exp(self._data))
