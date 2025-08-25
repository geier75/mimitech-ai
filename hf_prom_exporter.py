import argparse, json, os, time
from prometheus_client import Gauge, Histogram, start_http_server

def read_state(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def main():
    import signal, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--port", type=int, default=9108)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    state_path = os.path.join(args.output_dir, "trainer_state.json")
    start_http_server(args.port)

    g_loss = Gauge("mimikcompute_train_loss", "Training loss")
    g_lr   = Gauge("mimikcompute_train_lr", "Learning rate")
    g_gn   = Gauge("mimikcompute_train_grad_norm", "Gradient norm")
    g_tps  = Gauge("mimikcompute_train_throughput_samples", "Samples/sec")
    h_step = Histogram("mimikcompute_train_step_seconds", "Step time (s)",
                       buckets=(0.5,1,2,3,5,10,float("inf")))

    last_t, last_step = None, None
    while True:
        st = read_state(state_path)
        if st and st.get("log_history"):
            log = st["log_history"][-1]
            if "loss" in log: g_loss.set(float(log["loss"]))
            if "learning_rate" in log: g_lr.set(float(log["learning_rate"]))
            if "grad_norm" in log: g_gn.set(float(log["grad_norm"]))
            if "samples_per_second" in log: g_tps.set(float(log["samples_per_second"]))
            step = log.get("step") or st.get("global_step")
            now = time.time()
            if last_t is not None and last_step is not None and step is not None:
                ds = int(step) - int(last_step)
                if ds > 0:
                    h_step.observe((now - last_t) / ds)
            last_t, last_step = now, step
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
