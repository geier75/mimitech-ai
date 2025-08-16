    def get_active_backend(self) -> str:
        """
        Gibt das aktuell aktive Backend zurück.
        
        Returns:
            Name des aktiven Backends ('mlx', 'torch', 'numpy')
        """
        if self.use_mlx and self.mlx_backend is not None:
            return 'mlx'
        elif self.device in ['cuda', 'mps']:
            return 'torch'
        else:
            # CPU kann entweder PyTorch oder NumPy sein; wir bevorzugen PyTorch
            return 'torch'
    
    def evaluate(self, expression: str) -> Any:
        """
        Evaluiert einen mathematischen Ausdruck.
        
        Args:
            expression: Mathematischer Ausdruck als String
            
        Returns:
            Ergebnis der Auswertung
        """
        logger.info(f"Evaluiere Ausdruck: {expression}")
        try:
            # Für einfache arithmetische Ausdrücke können wir eval verwenden
            # In einer echten Implementierung würde hier eine sichere Parser-Bibliothek verwendet
            # werden (wie z.B. sympy)
            
            # Sicherheitsmaßnahmen: Nur numerische Operationen erlauben
            allowed_names = {
                'abs': abs, 'max': max, 'min': min, 'pow': pow, 'round': round,
                'sum': sum, 'len': len, 'int': int, 'float': float,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'exp': math.exp, 'log': math.log, 'sqrt': math.sqrt,
                'pi': math.pi, 'e': math.e
            }
            
            # Verwende NumPy für komplexere mathematische Funktionen
            for name in dir(np):
                if name.startswith('__'):
                    continue
                if callable(getattr(np, name)) or isinstance(getattr(np, name), (int, float)):
                    allowed_names[name] = getattr(np, name)
            
            # Evaluiere den Ausdruck in einem eingeschränkten Kontext
            # Dies ist noch nicht vollständig sicher und sollte in einer produktiven
            # Umgebung durch eine spezialisierte mathematische Parser-Bibliothek ersetzt werden
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return result
        except Exception as e:
            logger.error(f"Fehler bei der Auswertung von '{expression}': {e}")
            return None
    
    def solve_equation(self, equation: str, variable: Optional[str] = None) -> Any:
        """
        Löst eine mathematische Gleichung.
        
        Args:
            equation: Gleichung als String (z.B. "x^2 + 2*x - 3 = 0")
            variable: Zu lösende Variable (z.B. "x")
            
        Returns:
            Lösung der Gleichung
        """
        logger.info(f"Löse Gleichung: {equation} für {variable}")
        try:
            # In einer realen Implementierung würde hier eine symbolische Algebra-Bibliothek wie sympy verwendet
            # Für diesen Prototyp geben wir eine simulierte Lösung zurück
            return f"Lösung für {variable} in {equation}"
        except Exception as e:
            logger.error(f"Fehler beim Lösen der Gleichung '{equation}': {e}")
            return None
