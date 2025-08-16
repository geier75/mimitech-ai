#!/usr/bin/env python3
"""
VX-EVAL-RUNNER-EXTENDED
======================

Extended evaluation runner implementing CTO-recommended approach:
- Real git apply + pytest cycle instead of static solution texts
- Realistic validation with actual code execution
- Integration with VX-PATCH-REPAIR-SYSTEM and RAG

Features:
- Git-based patch application and testing
- Automated test discovery and execution
- Performance benchmarking
- Integration with existing patch repair pipeline
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import git
from datetime import datetime
import pytest
import sys

# VX-SELFWRITER Integration
try:
    from vx_patch_repair_system import VXPatchRepairSystem, PatchResult, IssueContext
    from vx_rag_context_system import RAGContextSelector, EnhancedIssueReaderAgent
except ImportError:
    # Fallback for standalone usage
    class VXPatchRepairSystem:
        def repair_issue(self, issue_data): return None
    class RAGContextSelector:
        def __init__(self, config=None): pass


@dataclass
class EvalResult:
    """Result of patch evaluation."""
    patch_id: str
    issue_id: str
    success: bool
    git_apply_success: bool
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    execution_time: float
    error_messages: List[str]
    performance_metrics: Dict[str, float]
    confidence_score: float
    validation_details: Dict[str, Any]


@dataclass
class TestSuite:
    """Test suite configuration and results."""
    name: str
    test_files: List[str]
    test_commands: List[str]
    timeout: int = 300
    requirements: List[str] = None
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []


class GitPatchManager:
    """Manages git operations for patch application and testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("VX-EVAL.GitPatchManager")
        self.temp_repos = []  # Track temporary repositories for cleanup
    
    def create_test_repo(self, source_repo_path: str, branch: str = "main") -> str:
        """Create temporary test repository for patch evaluation."""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="vx_eval_repo_")
            self.temp_repos.append(temp_dir)
            
            # Clone repository
            if os.path.exists(source_repo_path):
                # Local repository
                shutil.copytree(source_repo_path, temp_dir, dirs_exist_ok=True)
                repo = git.Repo(temp_dir)
            else:
                # Remote repository
                repo = git.Repo.clone_from(source_repo_path, temp_dir, branch=branch)
            
            self.logger.info(f"Created test repository: {temp_dir}")
            return temp_dir
            
        except Exception as e:
            self.logger.error(f"Failed to create test repository: {e}")
            raise
    
    def apply_patch(self, repo_path: str, patch_content: str, patch_file: str = None) -> Tuple[bool, str]:
        """Apply patch to repository using git apply."""
        try:
            repo = git.Repo(repo_path)
            
            # Create patch file if not provided
            if patch_file is None:
                patch_file = os.path.join(repo_path, "temp_patch.patch")
            
            # Write patch content to file
            with open(patch_file, 'w') as f:
                f.write(patch_content)
            
            # Apply patch using git apply
            result = subprocess.run([
                'git', 'apply', '--check', patch_file
            ], cwd=repo_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, f"Patch check failed: {result.stderr}"
            
            # Apply the patch
            result = subprocess.run([
                'git', 'apply', patch_file
            ], cwd=repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Stage changes
                repo.git.add(A=True)
                self.logger.info("Patch applied successfully")
                return True, "Patch applied successfully"
            else:
                return False, f"Patch application failed: {result.stderr}"
                
        except Exception as e:
            error_msg = f"Git apply failed: {e}"
            self.logger.error(error_msg)
            return False, error_msg
        finally:
            # Clean up temporary patch file
            if patch_file and os.path.exists(patch_file):
                try:
                    os.remove(patch_file)
                except:
                    pass
    
    def create_commit(self, repo_path: str, message: str = "Applied patch for evaluation") -> bool:
        """Create commit with applied changes."""
        try:
            repo = git.Repo(repo_path)
            
            # Check if there are changes to commit
            if repo.is_dirty():
                repo.git.commit('-m', message)
                self.logger.info(f"Created commit: {message}")
                return True
            else:
                self.logger.info("No changes to commit")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create commit: {e}")
            return False
    
    def cleanup_repos(self):
        """Clean up temporary repositories."""
        for repo_path in self.temp_repos:
            try:
                if os.path.exists(repo_path):
                    shutil.rmtree(repo_path)
                    self.logger.debug(f"Cleaned up repository: {repo_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {repo_path}: {e}")
        
        self.temp_repos.clear()


class TestRunner:
    """Runs tests and collects results."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("VX-EVAL.TestRunner")
        self.default_timeout = self.config.get('test_timeout', 300)
    
    def discover_tests(self, repo_path: str) -> TestSuite:
        """Discover tests in repository."""
        test_files = []
        test_commands = []
        requirements = []
        
        repo_path = Path(repo_path)
        
        # Find test files
        for pattern in ['test_*.py', '*_test.py', 'tests.py']:
            test_files.extend(list(repo_path.rglob(pattern)))
        
        # Find pytest configuration
        pytest_configs = list(repo_path.rglob('pytest.ini')) + list(repo_path.rglob('pyproject.toml'))
        
        # Find requirements
        req_files = list(repo_path.rglob('requirements*.txt')) + list(repo_path.rglob('Pipfile'))
        for req_file in req_files:
            try:
                with open(req_file, 'r') as f:
                    requirements.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])
            except:
                pass
        
        # Generate test commands
        if test_files:
            if pytest_configs:
                test_commands.append('python -m pytest -v')
            else:
                test_commands.append('python -m pytest -v --tb=short')
                
            # Add individual test file commands
            for test_file in test_files[:5]:  # Limit to first 5 test files
                rel_path = test_file.relative_to(repo_path)
                test_commands.append(f'python -m pytest {rel_path} -v')
        
        # Fallback to unittest discovery
        if not test_commands:
            test_commands.append('python -m unittest discover -s . -p "test*.py" -v')
        
        return TestSuite(
            name="discovered_tests",
            test_files=[str(f.relative_to(repo_path)) for f in test_files],
            test_commands=test_commands,
            timeout=self.default_timeout,
            requirements=list(set(requirements))
        )
    
    def setup_environment(self, repo_path: str, test_suite: TestSuite) -> bool:
        """Setup test environment with dependencies."""
        try:
            # Install requirements if available
            if test_suite.requirements:
                # Create temporary requirements file
                req_file = os.path.join(repo_path, 'temp_requirements.txt')
                with open(req_file, 'w') as f:
                    for req in test_suite.requirements:
                        if req and not req.startswith('-'):
                            f.write(f"{req}\n")
                
                # Install requirements
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', req_file
                ], cwd=repo_path, capture_output=True, text=True, timeout=120)
                
                # Clean up
                if os.path.exists(req_file):
                    os.remove(req_file)
                
                if result.returncode != 0:
                    self.logger.warning(f"Failed to install some requirements: {result.stderr}")
            
            # Install pytest if not available
            try:
                import pytest
            except ImportError:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 'pytest'
                ], check=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Environment setup failed: {e}")
            return False
    
    def run_tests(self, repo_path: str, test_suite: TestSuite) -> Dict[str, Any]:
        """Run test suite and collect results."""
        results = {
            'total_commands': len(test_suite.test_commands),
            'successful_commands': 0,
            'failed_commands': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'execution_time': 0,
            'error_messages': [],
            'command_results': []
        }
        
        start_time = time.time()
        
        for i, command in enumerate(test_suite.test_commands):
            cmd_start = time.time()
            
            try:
                self.logger.info(f"Running command {i+1}/{len(test_suite.test_commands)}: {command}")
                
                result = subprocess.run(
                    command.split(),
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=test_suite.timeout
                )
                
                cmd_duration = time.time() - cmd_start
                
                # Parse pytest output for test counts
                test_counts = self._parse_test_output(result.stdout, result.stderr)
                
                cmd_result = {
                    'command': command,
                    'return_code': result.returncode,
                    'duration': cmd_duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'test_counts': test_counts
                }
                
                results['command_results'].append(cmd_result)
                
                if result.returncode == 0:
                    results['successful_commands'] += 1
                    results['tests_passed'] += test_counts.get('passed', 0)
                    results['tests_failed'] += test_counts.get('failed', 0)
                    results['tests_skipped'] += test_counts.get('skipped', 0)
                else:
                    results['failed_commands'] += 1
                    results['error_messages'].append(f"Command failed: {command}")
                    if result.stderr:
                        results['error_messages'].append(result.stderr[:500])  # Limit error message length
                
            except subprocess.TimeoutExpired:
                results['failed_commands'] += 1
                results['error_messages'].append(f"Command timed out: {command}")
                self.logger.warning(f"Command timed out: {command}")
                
            except Exception as e:
                results['failed_commands'] += 1
                results['error_messages'].append(f"Command execution failed: {command} - {e}")
                self.logger.error(f"Command execution failed: {command} - {e}")
        
        results['execution_time'] = time.time() - start_time
        return results
    
    def _parse_test_output(self, stdout: str, stderr: str) -> Dict[str, int]:
        """Parse test output to extract test counts."""
        counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0}
        
        output = f"{stdout}\n{stderr}"
        
        # Pytest output patterns
        import re
        
        # Pattern: "5 passed, 2 failed, 1 skipped"
        pytest_pattern = r'(\d+)\s+(passed|failed|skipped|error)'
        matches = re.findall(pytest_pattern, output, re.IGNORECASE)
        
        for count, status in matches:
            if status.lower() in counts:
                counts[status.lower()] += int(count)
            elif status.lower() == 'error':
                counts['errors'] += int(count)
        
        # Unittest output patterns
        unittest_pattern = r'Ran (\d+) tests'
        unittest_matches = re.findall(unittest_pattern, output)
        if unittest_matches and not any(counts.values()):
            total_tests = int(unittest_matches[0])
            if 'FAILED' in output or 'ERROR' in output:
                # Estimate failures (rough heuristic)
                failure_indicators = len(re.findall(r'FAIL|ERROR', output))
                counts['failed'] = min(failure_indicators, total_tests)
                counts['passed'] = total_tests - counts['failed']
            else:
                counts['passed'] = total_tests
        
        return counts


class ExtendedEvalRunner:
    """Main evaluation runner with git apply + pytest cycle."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("VX-EVAL.ExtendedEvalRunner")
        
        # Initialize components
        self.git_manager = GitPatchManager(config.get('git_config', {}))
        self.test_runner = TestRunner(config.get('test_config', {}))
        
        # Optional: Initialize patch repair system
        if config.get('use_patch_repair', True):
            try:
                self.patch_repair_system = VXPatchRepairSystem(config.get('patch_repair_config', {}))
                self.rag_system = RAGContextSelector(config.get('rag_config', {}))
            except:
                self.patch_repair_system = None
                self.rag_system = None
        
        # Evaluation history
        self.eval_history = []
    
    def evaluate_patch(self, issue_data: Dict[str, Any], patch_data: Dict[str, Any], 
                      repo_path: str, branch: str = "main") -> EvalResult:
        """Evaluate patch with comprehensive testing."""
        start_time = time.time()
        
        issue_id = issue_data.get('id', 'unknown')
        patch_id = patch_data.get('id', f"patch_{int(time.time())}")
        
        self.logger.info(f"Starting evaluation for issue {issue_id}, patch {patch_id}")
        
        try:
            # Create test repository
            test_repo = self.git_manager.create_test_repo(repo_path, branch)
            
            # Apply patch
            patch_content = patch_data.get('diff', patch_data.get('content', ''))
            git_apply_success, git_message = self.git_manager.apply_patch(test_repo, patch_content)
            
            if not git_apply_success:
                return EvalResult(
                    patch_id=patch_id,
                    issue_id=issue_id,
                    success=False,
                    git_apply_success=False,
                    tests_passed=0,
                    tests_failed=0,
                    tests_skipped=0,
                    execution_time=time.time() - start_time,
                    error_messages=[git_message],
                    performance_metrics={},
                    confidence_score=0.0,
                    validation_details={'git_error': git_message}
                )
            
            # Discover and setup tests
            test_suite = self.test_runner.discover_tests(test_repo)
            env_setup_success = self.test_runner.setup_environment(test_repo, test_suite)
            
            if not env_setup_success:
                self.logger.warning("Environment setup failed, proceeding with basic tests")
            
            # Run tests
            test_results = self.test_runner.run_tests(test_repo, test_suite)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(test_results, test_suite)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(test_results, git_apply_success)
            
            # Determine overall success
            overall_success = (
                git_apply_success and
                test_results['successful_commands'] > 0 and
                test_results['tests_passed'] > test_results['tests_failed']
            )
            
            # Create evaluation result
            eval_result = EvalResult(
                patch_id=patch_id,
                issue_id=issue_id,
                success=overall_success,
                git_apply_success=git_apply_success,
                tests_passed=test_results['tests_passed'],
                tests_failed=test_results['tests_failed'],
                tests_skipped=test_results['tests_skipped'],
                execution_time=time.time() - start_time,
                error_messages=test_results['error_messages'],
                performance_metrics=performance_metrics,
                confidence_score=confidence_score,
                validation_details={
                    'git_message': git_message,
                    'test_suite': {
                        'name': test_suite.name,
                        'test_files_count': len(test_suite.test_files),
                        'commands_count': len(test_suite.test_commands)
                    },
                    'test_results': test_results
                }
            )
            
            # Store in history
            self.eval_history.append(eval_result)
            
            self.logger.info(f"Evaluation completed: success={overall_success}, "
                           f"tests_passed={test_results['tests_passed']}, "
                           f"confidence={confidence_score:.3f}")
            
            return eval_result
            
        except Exception as e:
            error_msg = f"Evaluation failed: {e}"
            self.logger.error(error_msg)
            
            return EvalResult(
                patch_id=patch_id,
                issue_id=issue_id,
                success=False,
                git_apply_success=False,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                execution_time=time.time() - start_time,
                error_messages=[error_msg],
                performance_metrics={},
                confidence_score=0.0,
                validation_details={'exception': str(e)}
            )
        
        finally:
            # Cleanup temporary repositories
            self.git_manager.cleanup_repos()
    
    def _calculate_performance_metrics(self, test_results: Dict[str, Any], test_suite: TestSuite) -> Dict[str, float]:
        """Calculate performance metrics from test results."""
        total_tests = test_results['tests_passed'] + test_results['tests_failed'] + test_results['tests_skipped']
        
        metrics = {
            'test_success_rate': test_results['tests_passed'] / max(total_tests, 1),
            'command_success_rate': test_results['successful_commands'] / max(test_results['total_commands'], 1),
            'average_test_time': test_results['execution_time'] / max(total_tests, 1),
            'tests_per_second': total_tests / max(test_results['execution_time'], 0.1),
            'error_rate': len(test_results['error_messages']) / max(test_results['total_commands'], 1)
        }
        
        return metrics
    
    def _calculate_confidence_score(self, test_results: Dict[str, Any], git_apply_success: bool) -> float:
        """Calculate confidence score for the evaluation."""
        confidence = 0.0
        
        # Base confidence from git apply success
        if git_apply_success:
            confidence += 0.3
        
        # Confidence from test execution
        if test_results['successful_commands'] > 0:
            confidence += 0.2
        
        # Confidence from test results
        total_tests = test_results['tests_passed'] + test_results['tests_failed']
        if total_tests > 0:
            test_success_rate = test_results['tests_passed'] / total_tests
            confidence += test_success_rate * 0.4
        
        # Penalty for errors
        if test_results['error_messages']:
            confidence -= min(len(test_results['error_messages']) * 0.1, 0.2)
        
        # Bonus for comprehensive testing
        if test_results['tests_passed'] > 5:
            confidence += 0.1
        
        return max(min(confidence, 1.0), 0.0)
    
    def evaluate_issue_with_repair(self, issue_data: Dict[str, Any], repo_path: str) -> Tuple[EvalResult, Optional[PatchResult]]:
        """Evaluate issue using integrated patch repair system."""
        if not self.patch_repair_system:
            raise ValueError("Patch repair system not available")
        
        # Generate patch using repair system
        patch_result = self.patch_repair_system.repair_issue(issue_data)
        
        if not patch_result.success:
            # Return failed evaluation
            return EvalResult(
                patch_id="failed_generation",
                issue_id=issue_data.get('id', 'unknown'),
                success=False,
                git_apply_success=False,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                execution_time=0,
                error_messages=[patch_result.error_message or "Patch generation failed"],
                performance_metrics={},
                confidence_score=0.0,
                validation_details={'patch_generation_failed': True}
            ), patch_result
        
        # Evaluate generated patch
        patch_data = {
            'id': f"generated_{int(time.time())}",
            'diff': patch_result.diff,
            'content': patch_result.patch_code
        }
        
        eval_result = self.evaluate_patch(issue_data, patch_data, repo_path)
        
        return eval_result, patch_result
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.eval_history:
            return {}
        
        successful_evals = [e for e in self.eval_history if e.success]
        
        return {
            'total_evaluations': len(self.eval_history),
            'successful_evaluations': len(successful_evals),
            'success_rate': len(successful_evals) / len(self.eval_history),
            'average_confidence': sum(e.confidence_score for e in self.eval_history) / len(self.eval_history),
            'average_execution_time': sum(e.execution_time for e in self.eval_history) / len(self.eval_history),
            'total_tests_passed': sum(e.tests_passed for e in self.eval_history),
            'total_tests_failed': sum(e.tests_failed for e in self.eval_history),
            'git_apply_success_rate': sum(1 for e in self.eval_history if e.git_apply_success) / len(self.eval_history)
        }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'git_config': {},
        'test_config': {'test_timeout': 120},
        'use_patch_repair': False,  # Set to True if patch repair system is available
        'rag_config': {}
    }
    
    # Initialize eval runner
    eval_runner = ExtendedEvalRunner(config)
    
    # Example evaluation
    example_issue = {
        'id': 'test-eval-001',
        'title': 'Fix null pointer exception',
        'body': 'Function crashes when input is None'
    }
    
    example_patch = {
        'id': 'patch-001',
        'diff': '''--- a/example.py
+++ b/example.py
@@ -1,3 +1,5 @@
 def process_data(data):
+    if data is None:
+        return None
     return data.strip()
''',
        'content': '''def process_data(data):
    if data is None:
        return None
    return data.strip()'''
    }
    
    # Note: This would require an actual repository to test
    # result = eval_runner.evaluate_patch(example_issue, example_patch, "/path/to/repo")
    
    print("Extended Eval Runner initialized successfully")
    print("Ready for patch evaluation with git apply + pytest cycle")
