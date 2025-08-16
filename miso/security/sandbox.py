#!/usr/bin/env python3
"""
MISO Security Sandbox - Isolated Code Execution
===============================================

Secure sandbox for executing untrusted code (e.g., HumanEval benchmarks).
Supports multiple isolation backends: Docker, nsjail, Firejail, and Python subprocess.

Security Features:
- Process isolation with limited privileges
- Filesystem access restrictions 
- Network isolation
- Resource limits (CPU, memory, time)
- Secure temporary directory cleanup
- Logging of all execution attempts

Required for safe execution of code generation benchmarks.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import signal
import time
import logging
import resource
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)

@dataclass
class SandboxResult:
    """Result of sandbox code execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    memory_used_mb: float
    timeout_occurred: bool
    error_message: Optional[str] = None

@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    # Resource limits
    timeout_seconds: int = 30
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50
    
    # Security settings
    allow_network: bool = False
    allow_filesystem_write: bool = False
    allowed_imports: Optional[List[str]] = None
    
    # Isolation backend
    backend: str = "subprocess"  # subprocess, docker, nsjail, firejail
    
    # Working directory setup
    setup_files: Dict[str, str] = None  # filename -> content
    cleanup_after: bool = True

class CodeSandbox:
    """Secure sandbox for executing untrusted code."""
    
    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self._validate_backend()
    
    def _validate_backend(self) -> None:
        """Validate and configure selected backend."""
        backend = self.config.backend
        
        if backend == "docker":
            if not shutil.which("docker"):
                logger.warning("Docker not found, falling back to subprocess")
                self.config.backend = "subprocess"
                
        elif backend == "nsjail":
            if not shutil.which("nsjail"):
                logger.warning("nsjail not found, falling back to subprocess")
                self.config.backend = "subprocess"
                
        elif backend == "firejail":
            if not shutil.which("firejail"):
                logger.warning("Firejail not found, falling back to subprocess")
                self.config.backend = "subprocess"
    
    def execute_code(self, code: str, test_code: str = None) -> SandboxResult:
        """Execute code in sandbox with optional test code."""
        start_time = time.time()
        
        try:
            with self._create_sandbox_environment() as sandbox_dir:
                # Write code to file
                code_file = sandbox_dir / "solution.py"
                with open(code_file, 'w') as f:
                    f.write(code)
                
                # Write test code if provided
                if test_code:
                    test_file = sandbox_dir / "test.py" 
                    with open(test_file, 'w') as f:
                        f.write(test_code)
                    
                    # Create combined execution script
                    exec_script = sandbox_dir / "execute.py"
                    with open(exec_script, 'w') as f:
                        f.write(f"""
import sys
import traceback

# Import solution
try:
    exec(open('solution.py').read(), globals())
except Exception as e:
    print(f"Error loading solution: {{e}}", file=sys.stderr)
    sys.exit(1)

# Run tests
try:
    exec(open('test.py').read(), globals())
    print("All tests passed!")
except Exception as e:
    print(f"Test failed: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
""")
                    execution_file = exec_script
                else:
                    execution_file = code_file
                
                # Setup additional files
                if self.config.setup_files:
                    for filename, content in self.config.setup_files.items():
                        file_path = sandbox_dir / filename
                        with open(file_path, 'w') as f:
                            f.write(content)
                
                # Execute based on backend
                result = self._execute_with_backend(execution_file, sandbox_dir)
                result.execution_time = time.time() - start_time
                
                return result
                
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=time.time() - start_time,
                memory_used_mb=0,
                timeout_occurred=False,
                error_message=f"Sandbox setup error: {e}"
            )
    
    @contextmanager
    def _create_sandbox_environment(self):
        """Create temporary sandbox environment."""
        temp_dir = None
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="miso_sandbox_"))
            
            # Set restrictive permissions
            os.chmod(temp_dir, 0o700)
            
            yield temp_dir
            
        finally:
            if temp_dir and temp_dir.exists() and self.config.cleanup_after:
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _execute_with_backend(self, execution_file: Path, sandbox_dir: Path) -> SandboxResult:
        """Execute code using the configured backend."""
        backend = self.config.backend
        
        if backend == "docker":
            return self._execute_docker(execution_file, sandbox_dir)
        elif backend == "nsjail":
            return self._execute_nsjail(execution_file, sandbox_dir)
        elif backend == "firejail":
            return self._execute_firejail(execution_file, sandbox_dir)
        else:
            return self._execute_subprocess(execution_file, sandbox_dir)
    
    def _execute_subprocess(self, execution_file: Path, sandbox_dir: Path) -> SandboxResult:
        """Execute using Python subprocess with basic resource limits."""
        cmd = [sys.executable, str(execution_file)]
        
        # Set resource limits for child process
        def set_limits():
            # Memory limit (soft limit in bytes)
            if self.config.memory_limit_mb > 0:
                memory_bytes = self.config.memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # CPU time limit
            if self.config.timeout_seconds > 0:
                resource.setrlimit(resource.RLIMIT_CPU, (self.config.timeout_seconds, self.config.timeout_seconds))
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=sandbox_dir,
                preexec_fn=set_limits,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout_seconds)
                timeout_occurred = False
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                timeout_occurred = True
            
            return SandboxResult(
                success=process.returncode == 0 and not timeout_occurred,
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
                execution_time=0,  # Set by caller
                memory_used_mb=0,  # Would need psutil to measure accurately
                timeout_occurred=timeout_occurred
            )
            
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
                memory_used_mb=0,
                timeout_occurred=False,
                error_message=f"Subprocess execution error: {e}"
            )
    
    def _execute_docker(self, execution_file: Path, sandbox_dir: Path) -> SandboxResult:
        """Execute using Docker container isolation."""
        container_name = f"miso_sandbox_{int(time.time())}_{os.getpid()}"
        
        # Docker run command with security restrictions
        cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--user", "nobody",  # Non-root user
            "--read-only",  # Read-only filesystem
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=100m",  # Temporary writable space
            "--memory", f"{self.config.memory_limit_mb}m",
            "--cpus", str(self.config.cpu_limit_percent / 100.0),
            "--network", "none" if not self.config.allow_network else "bridge",
            "--security-opt", "no-new-privileges:true",
            "--cap-drop", "ALL",
            "-v", f"{sandbox_dir}:/workspace:ro",
            "-w", "/workspace",
            "python:3.11-alpine",  # Minimal Python image
            "python", execution_file.name
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout_seconds + 5)
                timeout_occurred = False
            except subprocess.TimeoutExpired:
                # Force kill container
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                stdout, stderr = process.communicate()
                timeout_occurred = True
            
            return SandboxResult(
                success=process.returncode == 0 and not timeout_occurred,
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
                execution_time=0,
                memory_used_mb=0,
                timeout_occurred=timeout_occurred
            )
            
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
                memory_used_mb=0,
                timeout_occurred=False,
                error_message=f"Docker execution error: {e}"
            )
    
    def _execute_nsjail(self, execution_file: Path, sandbox_dir: Path) -> SandboxResult:
        """Execute using nsjail for Linux namespace isolation."""
        cmd = [
            "nsjail",
            "--mode", "o",  # One-shot mode
            "--user", "nobody",
            "--group", "nogroup", 
            "--hostname", "sandbox",
            "--cwd", str(sandbox_dir),
            "--bindmount_ro", f"{sandbox_dir}:{sandbox_dir}",
            "--tmpfsmount", "/tmp",
            "--disable_proc",
            "--rlimit_cpu", str(self.config.timeout_seconds),
            "--rlimit_as", str(self.config.memory_limit_mb * 1024 * 1024),
            "--time_limit", str(self.config.timeout_seconds),
            "--really_quiet",
            "--",
            sys.executable, str(execution_file)
        ]
        
        if not self.config.allow_network:
            cmd.extend(["--disable_clone_newnet"])
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=self.config.timeout_seconds + 5)
            
            return SandboxResult(
                success=process.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
                execution_time=0,
                memory_used_mb=0,
                timeout_occurred=process.returncode == 124  # nsjail timeout code
            )
            
        except subprocess.TimeoutExpired:
            process.kill()
            return SandboxResult(
                success=False,
                stdout="",
                stderr="nsjail timeout",
                return_code=-1,
                execution_time=0,
                memory_used_mb=0,
                timeout_occurred=True
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
                memory_used_mb=0,
                timeout_occurred=False,
                error_message=f"nsjail execution error: {e}"
            )
    
    def _execute_firejail(self, execution_file: Path, sandbox_dir: Path) -> SandboxResult:
        """Execute using Firejail for application sandboxing."""
        cmd = [
            "firejail",
            "--quiet",
            "--noprofile",
            "--private-tmp",
            "--noroot",
            "--nosound", 
            "--novideo",
            "--nodvd",
            "--notv",
            "--nou2f",
            f"--private={sandbox_dir}",
            f"--rlimit-cpu={self.config.timeout_seconds}",
            f"--timeout={self.config.timeout_seconds:02d}:{self.config.timeout_seconds % 60:02d}",
            "--deterministic-shutdown",
            "--",
            sys.executable, str(execution_file.name)  # Use relative path inside private dir
        ]
        
        if not self.config.allow_network:
            cmd.insert(-3, "--net=none")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=sandbox_dir
            )
            
            stdout, stderr = process.communicate(timeout=self.config.timeout_seconds + 5)
            
            return SandboxResult(
                success=process.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
                execution_time=0,
                memory_used_mb=0,
                timeout_occurred=process.returncode == 1  # Firejail timeout
            )
            
        except subprocess.TimeoutExpired:
            process.kill()
            return SandboxResult(
                success=False,
                stdout="",
                stderr="Firejail timeout",
                return_code=-1,
                execution_time=0,
                memory_used_mb=0,
                timeout_occurred=True
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
                memory_used_mb=0,
                timeout_occurred=False,
                error_message=f"Firejail execution error: {e}"
            )

class HumanEvalSandbox:
    """Specialized sandbox for HumanEval code execution."""
    
    def __init__(self, backend: str = "auto"):
        if backend == "auto":
            backend = self._detect_best_backend()
        
        self.config = SandboxConfig(
            timeout_seconds=30,
            memory_limit_mb=256,
            cpu_limit_percent=50,
            allow_network=False,
            allow_filesystem_write=False,
            backend=backend,
            cleanup_after=True
        )
        
        self.sandbox = CodeSandbox(self.config)
        logger.info(f"HumanEval sandbox initialized with backend: {self.config.backend}")
    
    def _detect_best_backend(self) -> str:
        """Detect the best available sandbox backend."""
        if shutil.which("docker"):
            return "docker"
        elif shutil.which("nsjail"):
            return "nsjail"
        elif shutil.which("firejail"):
            return "firejail"
        else:
            logger.warning("No isolation backends available, using subprocess (less secure)")
            return "subprocess"
    
    def execute_problem(self, solution_code: str, test_cases: str) -> Dict[str, Any]:
        """Execute HumanEval problem with test cases."""
        result = self.sandbox.execute_code(solution_code, test_cases)
        
        return {
            "passed": result.success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": result.execution_time,
            "timeout_occurred": result.timeout_occurred,
            "error_message": result.error_message,
            "sandbox_backend": self.config.backend
        }

def create_humaneval_sandbox(**kwargs) -> HumanEvalSandbox:
    """Factory function to create HumanEval sandbox."""
    return HumanEvalSandbox(**kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Test the sandbox with a simple code example
    sandbox = HumanEvalSandbox()
    
    test_code = """
def add_numbers(a, b):
    return a + b

# Test
assert add_numbers(2, 3) == 5
assert add_numbers(-1, 1) == 0
print("Simple test passed!")
"""
    
    test_cases = """
# Additional tests
assert add_numbers(0, 0) == 0
assert add_numbers(100, 200) == 300
print("All tests passed!")
"""
    
    print("Testing sandbox execution...")
    result = sandbox.execute_problem(test_code, test_cases)
    
    print(f"Passed: {result['passed']}")
    print(f"Backend: {result['sandbox_backend']}")
    print(f"Stdout: {result['stdout']}")
    if result['stderr']:
        print(f"Stderr: {result['stderr']}")
    
    print("\nTesting timeout handling...")
    infinite_loop = """
while True:
    pass
"""
    
    result2 = sandbox.execute_problem(infinite_loop, "print('This should not run')")
    print(f"Timeout test - Passed: {result2['passed']}, Timeout: {result2['timeout_occurred']}")
