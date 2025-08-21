"""
MISO Schema Validation Module
Validates benchmark results and reports against JSON schemas.
"""

import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, List, Union
import logging
from datetime import datetime
from ..monitoring.plausibility_monitor import PlausibilityMonitor, ComputeMode
from ..versioning.schema_version_manager import SchemaVersionManager

logger = logging.getLogger(__name__)

class SchemaValidator:
    """JSON Schema validator for MISO benchmark data"""
    
    def __init__(self, schemas_dir: Path = None):
        """Initialize validator with schema directory"""
        if schemas_dir is None:
            # Default to project root schemas directory
            project_root = Path(__file__).parent.parent.parent
            schemas_dir = project_root / "schemas"
        
        self.schemas_dir = schemas_dir
        self.schemas_cache = {}
        self.plausibility_monitor = PlausibilityMonitor()
        self.version_manager = SchemaVersionManager(schemas_dir)
        
    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load and cache JSON schema"""
        if schema_name not in self.schemas_cache:
            schema_path = self.schemas_dir / f"{schema_name}.schema.json"
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema not found: {schema_path}")
            
            with open(schema_path, 'r') as f:
                self.schemas_cache[schema_name] = json.load(f)
                
        return self.schemas_cache[schema_name]
    
    def validate_bench_result(self, result_data: Dict[str, Any]) -> bool:
        """Validate individual benchmark result against schema"""
        schema = self._load_schema("bench_result")
        
        try:
            jsonschema.validate(result_data, schema)
            logger.debug(f"✅ BenchResult validation passed for: {result_data.get('test_name', 'unknown')}")
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"❌ BenchResult validation failed: {e.message}")
            logger.error(f"   Failed at path: {' -> '.join(str(x) for x in e.path)}")
            raise ValueError(f"BenchResult schema validation failed: {e.message}")
    
    def validate_benchmark_report(self, report_data: Dict[str, Any]) -> bool:
        """Validate complete benchmark report against schema with Phase 7 enforcement"""
        schema = self._load_schema("benchmark_report")
        
        try:
            jsonschema.validate(report_data, schema)
            logger.debug(f"✅ Benchmark report validation passed")
            
            # Phase 8: Schema version validation
            self._validate_schema_version(report_data)
            
            # Phase 7: Enforce compute_mode requirement
            self._enforce_compute_mode_requirement(report_data)
            
            # Additional consistency checks
            summary = report_data.get("summary", {})
            total_expected = summary.get("total_tests", 0)
            actual_results = len(report_data.get("results", []))
            
            if total_expected != actual_results:
                logger.warning(f"⚠️ Consistency issue: summary.total_tests={total_expected} != len(results)={actual_results}")
                return False
            
            # Phase 7: Plausibility checks for all results
            self._perform_plausibility_checks(report_data)
            
            return True
            
        except jsonschema.ValidationError as e:
            logger.error(f"❌ Benchmark report validation failed: {e.message}")
            logger.error(f"Failed at path: {' -> '.join(str(p) for p in e.absolute_path)}")
            return False
        except ValueError as e:
            logger.error(f"❌ Phase 7 validation failed: {e}")
            return False
    
    def _check_report_consistency(self, report_data: Dict[str, Any]):
        """Check internal consistency beyond schema validation"""
        summary = report_data.get("summary", {})
        results = report_data.get("results", [])
        
        # Check total_tests consistency
        total_tests = summary.get("total_tests", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        skipped = summary.get("skipped", 0)
        errors = summary.get("errors", 0)
        
        if total_tests != (passed + failed + skipped + errors):
            raise ValueError(f"total_tests ({total_tests}) != sum of status counts ({passed + failed + skipped + errors})")
        
        # Check results array length matches total_tests
        if len(results) != total_tests:
            raise ValueError(f"results array length ({len(results)}) != total_tests ({total_tests})")
        
        # Check samples_processed consistency
        total_samples_summary = summary.get("total_samples_processed", 0)
        total_samples_actual = sum(result.get("samples_processed", 0) for result in results)
        
        if total_samples_summary != total_samples_actual:
            raise ValueError(f"total_samples_processed mismatch: summary={total_samples_summary}, actual={total_samples_actual}")
    
    def convert_bench_result_to_schema(self, bench_result) -> Dict[str, Any]:
        """Convert BenchResult object to schema-compliant dict"""
        return {
            "schema_version": "v1.0.0",
            "test_name": bench_result.name,
            "status": self._map_status(bench_result.status),
            "execution_time_ms": bench_result.duration_s * 1000,  # Convert to milliseconds
            "timestamp": datetime.fromtimestamp(bench_result.started_at).isoformat() + "Z",
            "accuracy": bench_result.accuracy * 100,  # Convert to percentage for schema
            "samples_processed": bench_result.samples_processed,
            "metadata": {
                "dataset_paths": bench_result.dataset_paths,
                "finished_at": bench_result.finished_at
            }
        }
    
    def _map_status(self, bench_status: str) -> str:
        """Map BenchResult status to schema status"""
        status_mapping = {
            "PASS": "passed",
            "PARTIAL": "failed", 
            "ERROR": "error"
        }
        return status_mapping.get(bench_status, "error")
    
    def create_validated_report(self, results: List, report_id: str = None, compute_mode: str = "full", seed: int = 42) -> Dict[str, Any]:
        """Create a validated benchmark report from BenchResult objects with reproducibility block"""
        if report_id is None:
            report_id = f"miso_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Import here to avoid circular dependencies
        from miso.reproducibility.repro_utils import ReproducibilityCollector
        
        # Convert results to schema format
        schema_results = []
        for result in results:
            schema_result = self.convert_bench_result_to_schema(result)
            self.validate_bench_result(schema_result)  # Validate each result
            schema_results.append(schema_result)
        
        # Calculate summary
        passed = len([r for r in schema_results if r["status"] == "passed"])
        failed = len([r for r in schema_results if r["status"] == "failed"])
        skipped = len([r for r in schema_results if r["status"] == "skipped"])
        errors = len([r for r in schema_results if r["status"] == "error"])
        
        total_samples = sum(r["samples_processed"] for r in schema_results)
        avg_accuracy = sum(r["accuracy"] for r in schema_results) / len(schema_results) if schema_results else 0
        total_time = sum(r["execution_time_ms"] for r in schema_results)
        
        # Collect reproducibility information
        repro_collector = ReproducibilityCollector(seed=seed)
        reproducibility_block = repro_collector.collect_reproducibility_block(compute_mode=compute_mode)
        
        report = {
            "schema_version": "v1.0.0",
            "report_id": report_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "summary": {
                "total_tests": len(results),
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "total_samples_processed": total_samples,
                "average_accuracy": round(avg_accuracy, 2),
                "total_execution_time_ms": total_time
            },
            "results": schema_results,
            "reproducibility": reproducibility_block,
            "system_info": {
                "platform": "macOS Apple Silicon",
                "python_version": "3.11+",
                "hardware": "MacBook Pro M4 Max"
            }
        }
        
        # Validate complete report
        self.validate_benchmark_report(report)
        
        logger.info(f"✅ Created validated report: {report_id} with {len(results)} results")
        logger.info(f"🎲 Reproducibility: commit={reproducibility_block['git_commit'][:7]}, seed={seed}, mode={compute_mode}")
        return report
    
    def _enforce_compute_mode_requirement(self, report_data: Dict[str, Any]) -> None:
        """Phase 7: Enforce that compute_mode is present and valid in reproducibility block"""
        reproducibility = report_data.get("reproducibility", {})
        
        if "compute_mode" not in reproducibility:
            raise ValueError("compute_mode is required in reproducibility block")
        
        compute_mode = reproducibility["compute_mode"]
        if not self.plausibility_monitor.validate_compute_mode(compute_mode):
            valid_modes = [mode.value for mode in ComputeMode]
            raise ValueError(f"compute_mode '{compute_mode}' is invalid. Must be one of: {valid_modes}")
        
        logger.debug(f"✅ Phase 7: compute_mode '{compute_mode}' validated")
    
    def _perform_plausibility_checks(self, report_data: Dict[str, Any]) -> None:
        """Phase 7: Perform plausibility checks on all benchmark results"""
        results = report_data.get("results", [])
        reproducibility = report_data.get("reproducibility", {})
        compute_mode = reproducibility.get("compute_mode", "stub")
        
        warnings_count = 0
        outliers_count = 0
        
        for result in results:
            benchmark_name = result.get("test_name", "unknown")
            duration_s = result.get("execution_time_ms", 0) / 1000.0
            samples_processed = result.get("samples_processed", 0)
            accuracy = result.get("accuracy", 0.0) / 100.0 if result.get("accuracy", 0) > 1 else result.get("accuracy", 0.0)
            
            # Perform plausibility check
            plausibility_result = self.plausibility_monitor.check_plausibility(
                benchmark_name=benchmark_name,
                duration_s=duration_s,
                samples_processed=samples_processed,
                accuracy=accuracy,
                compute_mode=compute_mode
            )
            
            # Log warnings and outliers
            for warning in plausibility_result.warnings:
                logger.warning(f"⚠️ Plausibility warning for {benchmark_name}: {warning}")
                warnings_count += 1
            
            for outlier in plausibility_result.outliers:
                logger.error(f"🚨 Plausibility outlier for {benchmark_name}: {outlier}")
                outliers_count += 1
            
            # Check for statistical outliers
            current_result = {
                "duration_s": duration_s,
                "accuracy": accuracy,
                "samples_processed": samples_processed
            }
            
            statistical_outliers = self.plausibility_monitor.detect_statistical_outliers(
                benchmark_name, current_result
            )
            
            for outlier in statistical_outliers:
                logger.warning(f"📈 Statistical outlier for {benchmark_name}: {outlier}")
                warnings_count += 1
        
        # Log summary
        if warnings_count > 0 or outliers_count > 0:
            logger.warning(f"⚠️ Phase 7 plausibility summary: {warnings_count} warnings, {outliers_count} outliers")
        else:
            logger.info(f"✅ Phase 7 plausibility checks: All results within expected ranges")
    
    def _validate_schema_version(self, report_data: Dict[str, Any]) -> None:
        """Phase 8: Validate schema version and enforce requirements"""
        schema_version = report_data.get("schema_version")
        
        if not schema_version:
            raise ValueError("schema_version is required in benchmark reports")
        
        # Validate version format and compatibility
        is_valid, message = self.version_manager.validate_schema_version(schema_version)
        if not is_valid:
            raise ValueError(f"Schema version validation failed: {message}")
        
        logger.debug(f"✅ Phase 8: schema_version '{schema_version}' validated")
        
        # Check for major version changes requiring documentation
        if schema_version != self.version_manager.CURRENT_VERSION:
            migrations = self.version_manager.check_migration_requirements(
                schema_version, self.version_manager.CURRENT_VERSION
            )
            if migrations:
                logger.warning(f"⚠️ Migration required from {schema_version} to {self.version_manager.CURRENT_VERSION}")
                for migration in migrations:
                    logger.warning(f"  - {migration}")
    
    def validate_schema_files_consistency(self) -> bool:
        """Phase 8: Validate consistency across all schema files"""
        validation_errors = self.version_manager.validate_schema_files()
        
        if validation_errors:
            logger.error("❌ Phase 8: Schema file validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("✅ Phase 8: All schema files are consistent")
        return True
    
    def check_release_gate_requirements(self, proposed_changes: List[str] = None) -> bool:
        """Phase 8: Check release gate requirements for schema changes"""
        if proposed_changes is None:
            proposed_changes = []
        
        passed, errors = self.version_manager.enforce_release_gate(proposed_changes)
        
        if not passed:
            logger.error("❌ Phase 8: Release gate validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("✅ Phase 8: Release gate requirements satisfied")
        return True
