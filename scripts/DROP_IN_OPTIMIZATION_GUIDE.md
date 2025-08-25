# 🚀 Drop-In Training Optimizations Guide

## Immediate Actions (No Training Interruption Required)

### 1. Add Prometheus Telemetry to Running Training

Add this to your existing `train_mps.py` **without stopping current run**:

```python
# Add this import at the top
from prometheus_callback import create_prometheus_callback

# In your trainer initialization, add:
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[create_prometheus_callback(port=9108)]  # Add this line
)
```

**Result**: Immediate telemetry at `http://localhost:9108/metrics`

### 2. Update TrainingArguments (Next Restart Only)

Replace your current `TrainingArguments` with:

```python
training_args = TrainingArguments(
    output_dir=OUTDIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    warmup_steps=100,
    
    # ✨ NEW: Optimized evaluation and saving
    evaluation_strategy="steps",
    eval_steps=1000,                    # Evaluate every 1000 steps
    save_strategy="steps", 
    save_steps=1000,                    # Save every 1000 steps
    save_total_limit=2,                 # Keep only 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # ✨ NEW: Performance optimizations  
    logging_steps=50,
    max_grad_norm=1.0,                  # Gradient clipping
    fp16=False,  # MPS doesn't support fp16
    dataloader_pin_memory=False,
    remove_unused_columns=False
)
```

### 3. DataLoader Optimization (Next Restart)

Add these kwargs to your Trainer:

```python
# Optimized DataLoader settings
dl_kwargs = {
    "num_workers": 8,
    "pin_memory": True, 
    "persistent_workers": True,
    "prefetch_factor": 6
}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    **dl_kwargs  # Add this
)
```

## 🎯 Expected Performance Improvements

| Optimization | Expected Improvement | Implementation |
|-------------|---------------------|----------------|
| **Prometheus Telemetry** | Real-time monitoring | ✅ Immediate (no restart) |
| **Eval/Save Rhythm** | Better checkpoint quality | 🔄 Next restart |
| **DataLoader Tuning** | 10-20% throughput boost | 🔄 Next restart |
| **Gradient Clipping** | Training stability | 🔄 Next restart |
| **Save Limit=2** | Disk space efficiency | 🔄 Next restart |

## 📊 Monitoring Setup

### Grafana Dashboard Panels

Create these 3 essential panels:

1. **Training Loss** 
   - Metric: `mimikcompute_train_loss`
   - Type: Time series

2. **Step Performance P95**
   - Metric: `histogram_quantile(0.95, mimikcompute_train_step_seconds_bucket)`
   - Alert: If P95 > 2× median over 1h

3. **Throughput**
   - Metric: `mimikcompute_train_throughput_samples`
   - Alert: If < 70% of median over 1h

### Alerts Setup

```yaml
# Prometheus alerts.yml
groups:
- name: training_alerts
  rules:
  - alert: HighStepTime
    expr: histogram_quantile(0.95, rate(mimikcompute_train_step_seconds_bucket[5m])) > 2 * histogram_quantile(0.50, rate(mimikcompute_train_step_seconds_bucket[1h]))
    for: 2m
    annotations:
      summary: "Step time P95 exceeds 2x median"
      
  - alert: LowThroughput
    expr: mimikcompute_train_throughput_samples < 0.7 * quantile_over_time(0.5, mimikcompute_train_throughput_samples[1h])
    for: 5m
    annotations:
      summary: "Throughput below 70% of recent median"
      
  - alert: AbnormalLoss
    expr: mimikcompute_nan_loss_total > 0
    for: 0s
    annotations:
      summary: "NaN/Inf loss detected - training may need intervention"
```

## 🔧 Quick Integration Steps

### For Current Training (Minimal Disruption):

1. **Copy files to scripts/**:
   ```bash
   # Files are already created:
   # - prometheus_callback.py
   # - train_mps_enhanced.py
   ```

2. **Add telemetry to current run** (edit running script):
   ```python
   # Add this one line to existing trainer
   callbacks=[create_prometheus_callback(port=9108)]
   ```

3. **Start monitoring**:
   ```bash
   # Check telemetry
   curl http://localhost:9108/metrics | grep mimikcompute
   ```

### For Next Training Run:

1. **Use enhanced script**:
   ```bash
   python scripts/train_mps_enhanced.py
   ```

2. **Verify improvements**:
   - Evaluation every 1000 steps
   - Only 2 checkpoints kept  
   - Better DataLoader performance
   - Full telemetry dashboard

## 📋 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 9108 in use | Change port in callback: `PrometheusCallback(port=9109)` |
| High memory usage | Reduce `num_workers` to 4 |
| Slow DataLoader | Disable `pin_memory` on MPS |
| Training instability | Reduce `learning_rate` to 1e-4 |

## 🎯 Success Metrics

You'll know the optimizations are working when:

- ✅ Prometheus metrics available at `:9108/metrics`
- ✅ Step times consistent (~2.29s or better)
- ✅ Regular evaluation every 1000 steps
- ✅ Only 2 checkpoint folders in output directory
- ✅ No NaN/Inf losses in telemetry
- ✅ Throughput >= 70% of recent average

## 🔄 Migration Timeline

| Phase | Action | Downtime |
|-------|--------|----------|
| **Phase 1** | Add Prometheus callback | 0 minutes |
| **Phase 2** | Update training args for next run | 0 minutes |
| **Phase 3** | Use enhanced script | Normal restart |

**Total implementation time**: < 5 minutes active work
