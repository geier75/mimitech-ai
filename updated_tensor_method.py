    def tensor(self, data, dtype=None, device=None):
        """
        Erstellt einen Tensor aus den gegebenen Daten.
        
        Args:
            data: Eingabedaten (NumPy-Array, Liste, PyTorch-Tensor oder MLX-Array)
            dtype: Optional, Datentyp für den Tensor
            device: Optional, Gerät für den Tensor
            
        Returns:
            Erstellter Tensor (MLXTensorWrapper oder TorchTensorWrapper)
        """
        from .tensor_wrappers import MLXTensorWrapper, TorchTensorWrapper
        
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
                return MLXTensorWrapper(mlx_tensor)
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
            return TorchTensorWrapper(tensor_data)
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung eines PyTorch-Tensors: {e}")
            raise
