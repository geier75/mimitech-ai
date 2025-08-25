#!/usr/bin/env python3
"""Grafana Dashboard Generator for MISO Training Telemetry"""

import json
from datetime import datetime

def create_training_dashboard():
    """Generate Grafana dashboard JSON for MISO training monitoring"""
    
    dashboard = {
        "dashboard": {
            "id": None,
            "title": "MISO Training Telemetry",
            "tags": ["miso", "training", "ml"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Training Loss",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "mimikcompute_train_loss",
                            "legendFormat": "Training Loss",
                            "refId": "A"
                        },
                        {
                            "expr": "mimikcompute_eval_loss", 
                            "legendFormat": "Validation Loss",
                            "refId": "B"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "unit": "none",
                            "decimals": 4
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                },
                {
                    "id": 2,
                    "title": "Step Performance (P95)",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(mimikcompute_train_step_seconds_bucket[5m]))",
                            "legendFormat": "P95 Step Time",
                            "refId": "A"
                        },
                        {
                            "expr": "histogram_quantile(0.50, rate(mimikcompute_train_step_seconds_bucket[5m]))",
                            "legendFormat": "Median Step Time", 
                            "refId": "B"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "unit": "s",
                            "decimals": 2
                        }
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "alert": {
                        "name": "High Step Time Alert",
                        "message": "Step time P95 exceeds 2x median",
                        "frequency": "10s",
                        "conditions": [
                            {
                                "query": {"queryType": "", "refId": "A"},
                                "reducer": {"type": "last", "params": []},
                                "evaluator": {"params": [5.0], "type": "gt"}
                            }
                        ]
                    }
                },
                {
                    "id": 3,
                    "title": "Training Throughput",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "mimikcompute_train_throughput_samples",
                            "legendFormat": "Samples/sec",
                            "refId": "A"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "unit": "ops",
                            "decimals": 1
                        }
                    },
                    "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8}
                },
                {
                    "id": 4,
                    "title": "Learning Rate",
                    "type": "timeseries", 
                    "targets": [
                        {
                            "expr": "mimikcompute_train_lr",
                            "legendFormat": "Learning Rate",
                            "refId": "A"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "unit": "scientific",
                            "decimals": 6
                        }
                    },
                    "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8}
                },
                {
                    "id": 5,
                    "title": "Gradient Norm",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": "mimikcompute_train_grad_norm",
                            "legendFormat": "Gradient Norm", 
                            "refId": "A"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "unit": "none",
                            "decimals": 3
                        }
                    },
                    "gridPos": {"h": 8, "w": 8, "x": 16, "y": 8}
                },
                {
                    "id": 6,
                    "title": "Training Progress",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "mimikcompute_train_step",
                            "legendFormat": "Current Step",
                            "refId": "A"
                        },
                        {
                            "expr": "mimikcompute_train_epoch",
                            "legendFormat": "Current Epoch",
                            "refId": "B"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "unit": "none",
                            "decimals": 0
                        }
                    },
                    "gridPos": {"h": 4, "w": 12, "x": 0, "y": 16}
                },
                {
                    "id": 7,
                    "title": "System Health",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "increase(mimikcompute_nan_loss_total[1h])",
                            "legendFormat": "NaN Losses (1h)",
                            "refId": "A"
                        },
                        {
                            "expr": "mimikcompute_checkpoints_total",
                            "legendFormat": "Checkpoints Saved",
                            "refId": "B"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "red", "value": 1}
                                ]
                            },
                            "unit": "none"
                        }
                    },
                    "gridPos": {"h": 4, "w": 12, "x": 12, "y": 16}
                }
            ],
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "5s",
            "schemaVersion": 27,
            "version": 0,
            "links": []
        },
        "meta": {
            "type": "db",
            "canSave": True,
            "canEdit": True,
            "canAdmin": True,
            "canStar": True,
            "slug": "miso-training-telemetry",
            "url": "/d/miso-training/miso-training-telemetry",
            "expires": "0001-01-01T00:00:00Z",
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "updatedBy": "admin",
            "createdBy": "admin",
            "version": 1,
            "hasAcl": False,
            "isFolder": False,
            "folderId": 0,
            "folderTitle": "General",
            "folderUrl": "",
            "provisioned": False,
            "provisionedExternalId": ""
        }
    }
    
    return dashboard

def create_alerts_config():
    """Generate Prometheus alerts configuration"""
    
    alerts = {
        "groups": [
            {
                "name": "miso_training_alerts",
                "rules": [
                    {
                        "alert": "HighStepTime",
                        "expr": "histogram_quantile(0.95, rate(mimikcompute_train_step_seconds_bucket[5m])) > 2 * histogram_quantile(0.50, rate(mimikcompute_train_step_seconds_bucket[1h]))",
                        "for": "2m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Training step time P95 exceeds 2x median",
                            "description": "Step execution is significantly slower than normal"
                        }
                    },
                    {
                        "alert": "LowThroughput", 
                        "expr": "mimikcompute_train_throughput_samples < 0.7 * quantile_over_time(0.5, mimikcompute_train_throughput_samples[1h])",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Training throughput below 70% of recent median",
                            "description": "Training performance has degraded significantly"
                        }
                    },
                    {
                        "alert": "AbnormalLoss",
                        "expr": "increase(mimikcompute_nan_loss_total[5m]) > 0",
                        "for": "0s",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "NaN or infinite loss detected",
                            "description": "Training has encountered numerical instability"
                        }
                    },
                    {
                        "alert": "HighGradientNorm",
                        "expr": "mimikcompute_train_grad_norm > 10.0",
                        "for": "1m", 
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Gradient norm is unusually high",
                            "description": "May indicate training instability"
                        }
                    },
                    {
                        "alert": "TrainingStalled",
                        "expr": "increase(mimikcompute_train_step[10m]) == 0",
                        "for": "10m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "Training appears to have stalled",
                            "description": "No progress detected in the last 10 minutes"
                        }
                    }
                ]
            }
        ]
    }
    
    return alerts

def main():
    """Generate and save dashboard and alerts configurations"""
    
    # Generate dashboard
    dashboard = create_training_dashboard()
    with open("miso_training_dashboard.json", "w") as f:
        json.dump(dashboard, f, indent=2)
    print("✅ Grafana dashboard saved to miso_training_dashboard.json")
    
    # Generate alerts (manual YAML format)
    alerts = create_alerts_config()
    with open("prometheus_alerts.yml", "w") as f:
        f.write("# MISO Training Alerts Configuration\n")
        f.write("groups:\n")
        f.write("- name: miso_training_alerts\n")
        f.write("  rules:\n")
        for rule in alerts["groups"][0]["rules"]:
            f.write(f"  - alert: {rule['alert']}\n")
            f.write(f"    expr: {rule['expr']}\n")
            f.write(f"    for: {rule['for']}\n")
            f.write(f"    labels:\n")
            for k, v in rule['labels'].items():
                f.write(f"      {k}: {v}\n")
            f.write(f"    annotations:\n")
            for k, v in rule['annotations'].items():
                f.write(f"      {k}: \"{v}\"\n")
            f.write("\n")
    print("✅ Prometheus alerts saved to prometheus_alerts.yml")
    
    print("\n🔧 Setup Instructions:")
    print("1. Import miso_training_dashboard.json into Grafana")
    print("2. Add prometheus_alerts.yml to your Prometheus config") 
    print("3. Configure Prometheus to scrape localhost:9108/metrics")
    print("4. Start training with PrometheusCallback enabled")
    print("\n📊 Dashboard will be available at: http://localhost:3000/d/miso-training/")

if __name__ == "__main__":
    main()
