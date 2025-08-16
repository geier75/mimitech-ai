    def tensor(self, data, dtype=None, device=None):
        """
        Erstellt einen Tensor aus den gegebenen Daten.
        
        Args:
            data: Eingabedaten (NumPy-Array, Liste, PyTorch-Tensor oder MLX-Array)
            dtype: Optional, Datentyp für den Tensor
            device: Optional, Gerät für den Tensor
            
        Returns:
            Erstellter Tensor
        """
        logger.info(f"Erstelle Tensor aus Daten mit Shape {getattr(data, 'shape', None)}")
        
        # Verwende MLX für Apple Silicon, falls verfügbar
        if self.use_mlx and self.mlx_backend is not None:
            try:
                # Konvertiere zu MLX-Array, falls nötig
                if isinstance(data, (np.ndarray, list, tuple)):
                    import mlx.core as mx
                    mlx_tensor = mx.array(data, dtype=self._get_mlx_dtype(dtype))
                elif hasattr(data, '__array__'):  # PyTorch-Tensor oder ähnliches
                    import mlx.core as mx
                    mlx_tensor = mx.array(data.detach().cpu().numpy() if hasattr(data, 'detach') else data.__array__(), 
                                         dtype=self._get_mlx_dtype(dtype))
                else:  # Bereits MLX-Array
                    mlx_tensor = data
                
                logger.info(f"MLX-Tensor erstellt mit Shape {mlx_tensor.shape}")
                return mlx_tensor
            except Exception as e:
                logger.warning(f"Fehler bei der Erstellung eines MLX-Tensors: {e}")
                # Fallback auf PyTorch
        
        # PyTorch als Fallback
        try:
            # Konvertiere zu PyTorch-Tensor, falls nötig
            if isinstance(data, (np.ndarray, list, tuple)):
                tensor_data = torch.tensor(data, dtype=self._get_torch_dtype(dtype))
            elif hasattr(data, 'detach'):  # PyTorch-Tensor
                tensor_data = data
                if dtype is not None:
                    tensor_data = tensor_data.to(dtype=self._get_torch_dtype(dtype))
            else:  # Anderer Tensor-Typ (z.B. MLX)
                if hasattr(data, '__array__'):
                    tensor_data = torch.tensor(data.__array__(), dtype=self._get_torch_dtype(dtype))
                else:
                    tensor_data = torch.tensor(np.array(data), dtype=self._get_torch_dtype(dtype))
            
            # Verschiebe auf das richtige Gerät
            target_device = device or self.device
            tensor_data = tensor_data.to(device=target_device)
            
            logger.info(f"PyTorch-Tensor erstellt mit Shape {tensor_data.shape} auf Gerät {target_device}")
            return tensor_data
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung eines PyTorch-Tensors: {e}")
            raise
    
    def _get_torch_dtype(self, dtype):
        """Konvertiert einen dtype-String in einen PyTorch-dtype"""
        if dtype is None:
            return torch.float32
        
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float64': torch.float64,
            'int32': torch.int32,
            'int64': torch.int64,
            'uint8': torch.uint8,
            'int8': torch.int8,
        }
        
        if isinstance(dtype, str):
            return dtype_map.get(dtype.lower(), torch.float32)
        return dtype  # Falls bereits ein torch.dtype Objekt
    
    def _get_mlx_dtype(self, dtype):
        """Konvertiert einen dtype-String in einen MLX-dtype"""
        if dtype is None:
            return None  # MLX-Standardtyp verwenden
        
        if not HAS_MLX:
            return None
            
        import mlx.core as mx
        
        dtype_map = {
            'float32': mx.float32,
            'float16': mx.float16,
            'bfloat16': mx.bfloat16,
            'int32': mx.int32,
            'int64': mx.int64,
            'uint8': mx.uint8,
            'int8': mx.int8,
        }
        
        if isinstance(dtype, str):
            return dtype_map.get(dtype.lower(), None)
        return dtype  # Falls bereits ein mlx.core.dtype Objekt
