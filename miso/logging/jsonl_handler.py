"""
JSONL logging handler for structured benchmark logs
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union


class JSONLHandler(logging.Handler):
    """Logging handler that writes JSON Lines format"""
    
    def __init__(self, filepath: Union[str, Path], mode: str = 'a', encoding: str = 'utf-8'):
        super().__init__()
        self.filepath = Path(filepath)
        self.mode = mode
        self.encoding = encoding
        
        # Ensure parent directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file handle
        self._file_handle: Optional[object] = None
        
    def emit(self, record: logging.LogRecord):
        """Emit a log record as a JSON line"""
        try:
            # Parse JSON from the message (structured logger sends JSON strings)
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                try:
                    # Try to parse as JSON first
                    log_data = json.loads(record.msg)
                except json.JSONDecodeError:
                    # Fall back to standard log record
                    log_data = {
                        'timestamp': self.formatTime(record),
                        'level': record.levelname,
                        'message': record.getMessage(),
                        'logger': record.name,
                        'module': record.module,
                        'funcName': record.funcName,
                        'lineno': record.lineno
                    }
            else:
                # Standard log record
                log_data = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'logger': record.name,
                    'module': record.module,
                    'funcName': record.funcName,
                    'lineno': record.lineno
                }
            
            # Write as JSON line
            self._write_line(json.dumps(log_data, ensure_ascii=False))
            
        except Exception as e:
            # Handle errors gracefully - don't break logging
            self.handleError(record)
    
    def _write_line(self, line: str):
        """Write a line to the JSONL file"""
        try:
            if self._file_handle is None:
                self._file_handle = open(self.filepath, self.mode, encoding=self.encoding)
            
            self._file_handle.write(line + '\n')
            self._file_handle.flush()
            
        except Exception as e:
            # If file writing fails, try to reopen
            self._reopen_file()
            if self._file_handle:
                self._file_handle.write(line + '\n')
                self._file_handle.flush()
    
    def _reopen_file(self):
        """Reopen the file handle"""
        try:
            if self._file_handle:
                self._file_handle.close()
        except:
            pass
        
        try:
            self._file_handle = open(self.filepath, self.mode, encoding=self.encoding)
        except Exception as e:
            self._file_handle = None
            raise
    
    def close(self):
        """Close the handler and file"""
        try:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
        finally:
            super().close()
    
    def formatTime(self, record: logging.LogRecord, datefmt: str = None) -> str:
        """Format timestamp for log records"""
        import datetime
        dt = datetime.datetime.fromtimestamp(record.created)
        return dt.isoformat() + "Z"


class JSONLFileRotator:
    """Utility to rotate JSONL log files by size or time"""
    
    @staticmethod
    def rotate_by_size(filepath: Path, max_size_mb: float = 10.0) -> bool:
        """Rotate log file if it exceeds size limit"""
        if not filepath.exists():
            return False
            
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            # Create backup with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = filepath.with_suffix(f".{timestamp}.jsonl")
            filepath.rename(backup_path)
            return True
        return False
    
    @staticmethod
    def rotate_by_time(filepath: Path, max_age_hours: float = 24.0) -> bool:
        """Rotate log file if it's older than max_age_hours"""
        if not filepath.exists():
            return False
            
        import datetime
        import time
        
        file_age_hours = (time.time() - filepath.stat().st_mtime) / 3600
        if file_age_hours > max_age_hours:
            # Create backup with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = filepath.with_suffix(f".{timestamp}.jsonl")
            filepath.rename(backup_path)
            return True
        return False
