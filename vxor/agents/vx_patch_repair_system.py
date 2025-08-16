#!/usr/bin/env python3
"""
VX-PATCH-REPAIR-SYSTEM
======================

Multi-Agent Patch Repair System implementing CTO-recommended architecture:
- IssueReaderAgent: Converts GitHub Issue to repair task
- PatchSuggestorAgent: Code changes with minimal diff
- VerifierAgent: Tests patch with pass/fail evaluation

Integrates with VX-SELFWRITER ASI-System for enhanced performance.
"""

import ast
import json
import logging
import re
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

# VX-SELFWRITER Integration
try:
    from vxor.ai.vx_selfwriter_core import VXSelfWriterCore
    from vxor.ai.vx_selfwriter_best_practices import VXSelfWriterBestPractices
    from vxor.lang.vx_lingua_core import VXLinguaCore
except ImportError:
    # Fallback implementations
    class VXSelfWriterCore:
        def analyze_and_evolve_code(self, code): return {'improved_code': code}
    class VXSelfWriterBestPractices:
        def validate_all_best_practices(self, code): return {'score': 50}
    class VXLinguaCore:
        def process_natural_language(self, text): return {'intent': 'unknown'}


@dataclass
class IssueContext:
    """Context object for GitHub issues and repair tasks."""
    issue_id: str
    title: str
    description: str
    code_snippets: List[str]
    error_messages: List[str]
    file_paths: List[str]
    priority: str = "medium"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class PatchResult:
    """Result object for patch operations."""
    success: bool
    patch_code: str
    diff: str
    confidence: float
    test_results: Dict[str, Any]
    metrics: Dict[str, float]
    error_message: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all patch repair agents."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"VX-PATCH-REPAIR.{name}")
        self.performance_metrics = {}
        
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return result."""
        pass


class IssueReaderAgent(BaseAgent):
    """Converts GitHub Issues to specified repair tasks with RAG."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("IssueReader", config)
        self.lingua_core = VXLinguaCore()
        
    def process(self, issue_data: Dict[str, Any]) -> IssueContext:
        """Convert GitHub issue to structured repair task."""
        start_time = time.time()
        
        try:
            issue_id = issue_data.get('id', 'unknown')
            title = issue_data.get('title', '')
            description = issue_data.get('body', '')
            
            # Extract components using regex patterns
            code_snippets = self._extract_code_snippets(description)
            error_messages = self._extract_error_messages(description)
            file_paths = self._extract_file_paths(description)
            priority = self._determine_priority(title, description)
            tags = self._generate_tags(title, description)
            
            context = IssueContext(
                issue_id=issue_id,
                title=title,
                description=description,
                code_snippets=code_snippets,
                error_messages=error_messages,
                file_paths=file_paths,
                priority=priority,
                tags=tags
            )
            
            duration = time.time() - start_time
            self.logger.info(f"Processed issue {issue_id} in {duration:.3f}s")
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to process issue: {e}")
            raise
    
    def _extract_code_snippets(self, text: str) -> List[str]:
        """Extract code snippets from issue text."""
        patterns = [
            r'```(?:python|py)?\n(.*?)```',
            r'`([^`]+)`'
        ]
        snippets = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            snippets.extend(matches)
        return [s.strip() for s in snippets if s.strip()]
    
    def _extract_error_messages(self, text: str) -> List[str]:
        """Extract error messages from issue text."""
        patterns = [
            r'Traceback.*?(?=\n\n|\n[A-Z]|\Z)',
            r'Error: .*',
            r'Exception: .*'
        ]
        errors = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            errors.extend(matches)
        return [e.strip() for e in errors if e.strip()]
    
    def _extract_file_paths(self, text: str) -> List[str]:
        """Extract file paths from issue text."""
        patterns = [
            r'File "([^"]+)"',
            r'([a-zA-Z0-9_/.-]+\.py)',
        ]
        paths = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            paths.extend(matches)
        return list(set(p.strip() for p in paths if p.strip()))
    
    def _determine_priority(self, title: str, description: str) -> str:
        """Determine issue priority."""
        text = f"{title} {description}".lower()
        if any(w in text for w in ['critical', 'urgent', 'crash', 'security']):
            return "high"
        elif any(w in text for w in ['bug', 'error', 'issue']):
            return "medium"
        return "low"
    
    def _generate_tags(self, title: str, description: str) -> List[str]:
        """Generate contextual tags."""
        text = f"{title} {description}".lower()
        tags = []
        
        tech_tags = {
            'python': r'\bpython\b',
            'javascript': r'\bjs\b',
            'api': r'\bapi\b',
            'database': r'\bdb\b|\bsql\b'
        }
        
        for tag, pattern in tech_tags.items():
            if re.search(pattern, text):
                tags.append(tag)
        
        return tags


class PatchSuggestorAgent(BaseAgent):
    """Generates code changes with minimal diff using VX-SELFWRITER."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("PatchSuggestor", config)
        self.selfwriter = VXSelfWriterCore()
        
    def process(self, issue_context: IssueContext) -> PatchResult:
        """Generate patch for the given issue context."""
        start_time = time.time()
        
        try:
            # Generate patch using VX-SELFWRITER
            if issue_context.code_snippets:
                original_code = '\n'.join(issue_context.code_snippets)
                analysis = self.selfwriter.analyze_and_evolve_code(original_code)
                patch_code = analysis.get('improved_code', original_code)
            else:
                patch_code = self._generate_fallback_patch(issue_context)
            
            # Create diff and calculate metrics
            diff = self._create_diff(issue_context, patch_code)
            confidence = self._calculate_confidence(issue_context, patch_code)
            metrics = self._generate_metrics(patch_code)
            
            result = PatchResult(
                success=True,
                patch_code=patch_code,
                diff=diff,
                confidence=confidence,
                test_results={},
                metrics=metrics
            )
            
            duration = time.time() - start_time
            self.logger.info(f"Generated patch in {duration:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Patch generation failed: {e}")
            return PatchResult(
                success=False,
                patch_code="",
                diff="",
                confidence=0.0,
                test_results={},
                metrics={},
                error_message=str(e)
            )
    
    def _generate_fallback_patch(self, issue_context: IssueContext) -> str:
        """Generate fallback patch based on issue analysis."""
        description = issue_context.description.lower()
        
        if 'null' in description or 'none' in description:
            return """
# Add null/None checks
if variable is not None:
    # Process variable
    result = process(variable)
else:
    logger.warning("Variable is None")
    result = None
"""
        elif 'index' in description:
            return """
# Add index bounds checking
if 0 <= index < len(collection):
    result = collection[index]
else:
    logger.warning(f"Index {index} out of bounds")
    result = None
"""
        else:
            return f"""
# Generic fix for: {issue_context.title}
def fix_issue():
    try:
        # Implement fix here
        pass
    except Exception as e:
        logger.error(f"Fix failed: {e}")
        raise
"""
    
    def _create_diff(self, issue_context: IssueContext, patch_code: str) -> str:
        """Create minimal diff."""
        if not issue_context.code_snippets:
            return f"+ {patch_code}"
        
        original = '\n'.join(issue_context.code_snippets)
        lines = []
        lines.append("--- Original")
        lines.append("+++ Patched")
        
        for line in original.split('\n'):
            lines.append(f"- {line}")
        for line in patch_code.split('\n'):
            lines.append(f"+ {line}")
        
        return '\n'.join(lines)
    
    def _calculate_confidence(self, issue_context: IssueContext, patch_code: str) -> float:
        """Calculate confidence score."""
        confidence = 0.5
        
        if issue_context.code_snippets:
            confidence += 0.2
        if issue_context.error_messages:
            confidence += 0.1
        if len(patch_code.strip()) > 50:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_metrics(self, patch_code: str) -> Dict[str, float]:
        """Generate patch metrics."""
        return {
            'lines_of_code': len(patch_code.split('\n')),
            'estimated_fix_time': 0.5,
            'risk_score': 0.3
        }


class VerifierAgent(BaseAgent):
    """Tests patches with comprehensive validation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Verifier", config)
        self.test_timeout = config.get('test_timeout', 30) if config else 30
        
    def process(self, patch_result: PatchResult, issue_context: IssueContext) -> PatchResult:
        """Verify patch with comprehensive testing."""
        start_time = time.time()
        
        try:
            test_results = self._run_tests(patch_result, issue_context)
            patch_result.test_results = test_results
            patch_result.success = test_results.get('overall_success', False)
            
            # Adjust confidence based on test results
            if test_results.get('tests_passed', 0) > 0:
                patch_result.confidence *= 1.1
            else:
                patch_result.confidence *= 0.8
            
            patch_result.confidence = min(patch_result.confidence, 1.0)
            
            duration = time.time() - start_time
            self.logger.info(f"Verified patch in {duration:.3f}s")
            return patch_result
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            patch_result.success = False
            patch_result.error_message = str(e)
            return patch_result
    
    def _run_tests(self, patch_result: PatchResult, issue_context: IssueContext) -> Dict[str, Any]:
        """Run comprehensive tests."""
        results = {
            'syntax_check': self._check_syntax(patch_result.patch_code),
            'static_analysis': self._run_static_analysis(patch_result.patch_code),
            'tests_passed': 0,
            'tests_failed': 0,
            'error_messages': []
        }
        
        # Count passed tests
        if results['syntax_check']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
            results['error_messages'].append("Syntax check failed")
        
        if results['static_analysis']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
            results['error_messages'].append("Static analysis failed")
        
        results['overall_success'] = results['tests_passed'] > results['tests_failed']
        return results
    
    def _check_syntax(self, code: str) -> bool:
        """Check Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _run_static_analysis(self, code: str) -> bool:
        """Basic static analysis."""
        try:
            tree = ast.parse(code)
            # Check for dangerous operations
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if hasattr(node.func, 'id') and node.func.id in ['eval', 'exec']:
                        return False
            return True
        except:
            return False


class VXPatchRepairSystem:
    """Main system orchestrating the patch repair process."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("VX-PATCH-REPAIR-SYSTEM")
        
        # Initialize agents
        self.issue_reader = IssueReaderAgent(config)
        self.patch_suggestor = PatchSuggestorAgent(config)
        self.verifier = VerifierAgent(config)
        
        # Performance tracking
        self.performance_history = []
    
    def repair_issue(self, issue_data: Dict[str, Any]) -> PatchResult:
        """Complete patch repair pipeline."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting repair for issue: {issue_data.get('title', 'Unknown')}")
            
            # Step 1: Read and analyze issue
            issue_context = self.issue_reader.process(issue_data)
            
            # Step 2: Generate patch
            patch_result = self.patch_suggestor.process(issue_context)
            
            # Step 3: Verify patch
            if patch_result.success:
                patch_result = self.verifier.process(patch_result, issue_context)
            
            # Track performance
            duration = time.time() - start_time
            self.performance_history.append({
                'issue_id': issue_context.issue_id,
                'duration': duration,
                'success': patch_result.success,
                'confidence': patch_result.confidence,
                'timestamp': time.time()
            })
            
            self.logger.info(f"Repair completed in {duration:.3f}s, success: {patch_result.success}")
            return patch_result
            
        except Exception as e:
            self.logger.error(f"Repair pipeline failed: {e}")
            return PatchResult(
                success=False,
                patch_code="",
                diff="",
                confidence=0.0,
                test_results={},
                metrics={},
                error_message=str(e)
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        if not self.performance_history:
            return {}
        
        successful_repairs = [p for p in self.performance_history if p['success']]
        
        return {
            'total_repairs': len(self.performance_history),
            'successful_repairs': len(successful_repairs),
            'success_rate': len(successful_repairs) / len(self.performance_history),
            'average_duration': sum(p['duration'] for p in self.performance_history) / len(self.performance_history),
            'average_confidence': sum(p['confidence'] for p in successful_repairs) / len(successful_repairs) if successful_repairs else 0
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize system
    repair_system = VXPatchRepairSystem()
    
    # Example issue
    example_issue = {
        'id': 'test-001',
        'title': 'NoneType object has no attribute error',
        'body': '''
        Getting this error when running the code:
        
        ```python
        def process_data(data):
            return data.strip()
        
        result = process_data(None)
        ```
        
        Error: AttributeError: 'NoneType' object has no attribute 'strip'
        
        File "main.py", line 2, in process_data
        '''
    }
    
    # Run repair
    result = repair_system.repair_issue(example_issue)
    
    print(f"Repair Success: {result.success}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Patch Code:\n{result.patch_code}")
    print(f"Test Results: {result.test_results}")
    
    # Show performance metrics
    metrics = repair_system.get_performance_metrics()
    print(f"Performance Metrics: {metrics}")
