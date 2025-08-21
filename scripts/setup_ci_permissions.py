#!/usr/bin/env python3
"""
Setup CI Permissions - Phase 13
Script to configure read-only access for CI environment
"""

import argparse
import sys
from pathlib import Path
from miso.governance.access_control import AccessController

def main():
    parser = argparse.ArgumentParser(description="Setup CI access permissions")
    parser.add_argument("--base-dir", 
                       help="Base directory for operations (default: current)")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify permissions, don't change them")
    parser.add_argument("--force", action="store_true",
                       help="Force setup even outside CI environment")
    
    args = parser.parse_args()
    
    # Initialize access controller
    controller = AccessController()
    
    base_dir = Path(args.base_dir) if args.base_dir else None
    
    # Check environment
    if not controller.is_ci_environment and not args.force:
        print("⚠️ Not in CI environment. Use --force to override.")
        if not args.verify_only:
            sys.exit(1)
    
    try:
        if args.verify_only:
            print("🔍 Verifying access control compliance...")
            
            # Generate compliance report
            compliance_report = controller.verify_access_compliance(base_dir)
            
            total_policies = len(compliance_report)
            compliant_policies = sum(1 for report in compliance_report.values() if report["compliant"])
            
            print(f"\n📊 Compliance Summary: {compliant_policies}/{total_policies} policies compliant")
            
            # Show violations
            for policy_name, report in compliance_report.items():
                if report["compliant"]:
                    print(f"  ✅ {policy_name}: Compliant")
                else:
                    print(f"  ❌ {policy_name}: {len(report['violations'])} violations")
                    for violation in report["violations"]:
                        print(f"    - {violation['path']}: expected {violation['expected']}")
            
            if compliant_policies < total_policies:
                print("\n💡 Run without --verify-only to fix permissions")
                sys.exit(1)
            else:
                print("\n✅ All access policies compliant")
                
        else:
            print("🔒 Setting up CI read-only permissions...")
            
            # Setup CI permissions
            success = controller.setup_ci_permissions(base_dir)
            
            if success:
                print("\n✅ CI permissions setup completed")
                
                # Verify the setup worked
                print("\n🔍 Verifying permissions...")
                compliance = controller.verify_access_compliance(base_dir)
                
                violations = sum(len(report["violations"]) for report in compliance.values())
                if violations == 0:
                    print("✅ All permissions verified successfully")
                else:
                    print(f"⚠️ {violations} permission violations remain")
                    sys.exit(1)
            else:
                print("\n❌ Failed to setup CI permissions")
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ Permission setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
