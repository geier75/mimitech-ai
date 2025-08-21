#!/usr/bin/env python3
"""
Phase 10: Standalone mutation test runner
Validates schema robustness against malformed inputs
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run comprehensive mutation tests for schema validation"""
    
    print("🧬 Running MISO mutation tests for schema robustness...")
    print("=" * 60)
    
    # Run the mutation test suite
    test_file = "tests/test_mutation_validation.py"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file, 
            "-v", 
            "--tb=short",
            "--color=yes"
        ], check=True, capture_output=False)
        
        print("\n" + "=" * 60)
        print("✅ All mutation tests passed - schema validation is robust!")
        print("\n📋 Mutation tests covered:")
        print("   • Missing required fields")
        print("   • Invalid accuracy ranges")  
        print("   • Invalid status values")
        print("   • Malformed JSON handling")
        print("   • Cross-check inconsistencies")
        print("   • Plausibility outlier detection")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Mutation tests failed with exit code {e.returncode}")
        print("Schema validation may have vulnerabilities that need addressing.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running mutation tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
