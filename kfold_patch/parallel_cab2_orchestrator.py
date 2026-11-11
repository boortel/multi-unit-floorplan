#!/usr/bin/env python3
"""
Parallel K-Fold Orchestrator for CAB2-V2 (Trial 32: EfficientNetV2M).

Manages parallel execution across GPUs:
- Tracks existing completed folds (Folds 0, 1, 2 completed; Fold 3 on GPU 3).
- GPU 2 worker runs pending folds in descending order (9 -> 8 -> 7 -> ...).
- Monitors GPU 3 (PID 2476618) running in ascending order (3 -> 4 -> ...).
- Prevents redundant duplicate runs on overlapping folds.
- Automatically triggers final evaluation (evaluate_kfold.py) when all 10 folds complete.
"""

import os
import sys
import time
import json
import glob
import fcntl
import signal
import subprocess
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(_PROJECT_ROOT, 'logs')
STATUS_FILE = os.path.join(LOGS_DIR, 'cab2_kfold_progress.json')
SUMMARY_FILE = os.path.join(LOGS_DIR, 'cab2_kfold_summary.txt')
LOCK_FILE = os.path.join(LOGS_DIR, 'cab2_kfold.lock')


def get_completed_folds():
    """Scans models/ directory to find all folds that have a valid saved_model.pb and training_cfg.py."""
    pattern = os.path.join(MODELS_DIR, 'cab2_cab2_hpo_trial32_v2m_EfficientNetV2M_*')
    run_dirs = sorted(glob.glob(pattern))
    completed = {}
    for d in run_dirs:
        if not os.path.isdir(d):
            continue
        for entry in os.listdir(d):
            if entry.isdigit():
                f_idx = int(entry)
                pb_path = os.path.join(d, entry, 'saved_model.pb')
                cfg_path = os.path.join(d, entry, 'training_cfg.py')
                if os.path.exists(pb_path) and os.path.getsize(pb_path) > 0 and os.path.exists(cfg_path):
                    completed[f_idx] = os.path.join(d, entry)
    return completed


def is_pid_alive(pid):
    """Check if process with given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def load_status():
    for _ in range(5):
        if os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                time.sleep(0.1)
        else:
            break
    return {}


def save_status(status):
    os.makedirs(LOGS_DIR, exist_ok=True)
    temp_file = f"{STATUS_FILE}.{os.getpid()}_{time.time_ns()}.tmp"
    for attempt in range(5):
        try:
            with open(temp_file, 'w') as f:
                json.dump(status, f, indent=2)
            os.replace(temp_file, STATUS_FILE)
            break
        except Exception as e:
            if attempt == 4:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass
                raise e
            time.sleep(0.2)


def update_overall_status():
    completed = get_completed_folds()
    status = load_status()

    fold_status = {}
    for f in range(10):
        if f in completed:
            fold_status[str(f)] = {
                'state': 'completed',
                'path': completed[f]
            }
        elif status.get('folds', {}).get(str(f), {}).get('state') == 'in_progress':
            old = status['folds'][str(f)]
            pid = old.get('pid')
            if pid and is_pid_alive(pid):
                fold_status[str(f)] = old
            else:
                fold_status[str(f)] = {'state': 'pending'}
        else:
            fold_status[str(f)] = {'state': 'pending'}

    status['updated_at'] = datetime.utcnow().isoformat()
    status['folds'] = fold_status
    status['completed_count'] = len(completed)
    save_status(status)

    # Write human-readable summary atomically
    summary_tmp = f"{SUMMARY_FILE}.{os.getpid()}_{time.time_ns()}.tmp"
    try:
        with open(summary_tmp, 'w') as out_f:
            out_f.write(f"=== CAB2-V2 K-Fold Parallel Status ({status['updated_at']}) ===\n")
            out_f.write(f"Completed: {len(completed)}/10 folds\n\n")
            for f in range(10):
                info = fold_status[str(f)]
                st = info['state']
                if st == 'completed':
                    out_f.write(f"  Fold {f}: [DONE] in {info.get('path', '')}\n")
                elif st == 'in_progress':
                    out_f.write(f"  Fold {f}: [IN PROGRESS] on GPU {info.get('gpu')} (PID {info.get('pid')})\n")
                else:
                    out_f.write(f"  Fold {f}: [PENDING]\n")
        os.replace(summary_tmp, SUMMARY_FILE)
    except Exception:
        if os.path.exists(summary_tmp):
            try:
                os.remove(summary_tmp)
            except OSError:
                pass
    return status


def claim_next_fold(gpu_id, order='asc'):
    """Atomically find and claim the next pending fold using file locking."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(LOCK_FILE, 'a') as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            status = update_overall_status()
            completed = get_completed_folds()
            if len(completed) >= 10:
                return None

            fold_range = list(range(10)) if order == 'asc' else list(reversed(range(10)))
            target_fold = None
            for f in fold_range:
                if f in completed:
                    continue
                f_state = status.get('folds', {}).get(str(f), {}).get('state', 'pending')
                if f_state == 'pending':
                    target_fold = f
                    break

            if target_fold is not None:
                status['folds'][str(target_fold)] = {
                    'state': 'in_progress',
                    'gpu': gpu_id,
                    'pid': os.getpid(),
                    'start_time': datetime.utcnow().isoformat()
                }
                save_status(status)
            return target_fold
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def run_worker_gpu(gpu_id=2, order='desc'):
    """Worker process that continuously picks the next pending fold and trains it."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting CAB2 worker on GPU {gpu_id} (order={order})...")

    while True:
        completed = get_completed_folds()
        if len(completed) >= 10:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All 10 folds completed! Worker exiting.")
            break

        target_fold = claim_next_fold(gpu_id, order=order)

        if target_fold is None:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No pending folds available for GPU {gpu_id}.")
            if len(get_completed_folds()) >= 10:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All 10 folds completed! Worker exiting.")
                break
            time.sleep(30)
            continue

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU {gpu_id} claimed Fold {target_fold}.")
        config_path = os.path.join(_PROJECT_ROOT, f'kfold_patch/eval_cab2_hpo_v2m_fold{target_fold}.py')
        if not os.path.exists(config_path):
            with open(config_path, 'w') as cfp:
                cfp.write(f"_base_ = ['eval_cab2_hpo_v2m.py']\ntarget_fold = {target_fold}\n")

        log_path = os.path.join(LOGS_DIR, f'kfold_cab2_fold{target_fold}_gpu{gpu_id}.log')

        cmd = [
            sys.executable,
            os.path.join(_PROJECT_ROOT, 'train_config.py'),
            config_path
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['PYTHONUNBUFFERED'] = '1'

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Launching: {' '.join(cmd)} -> {log_path}")
        with open(log_path, 'w') as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=_PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            # Update pid
            status = load_status()
            if str(target_fold) in status.get('folds', {}):
                status['folds'][str(target_fold)]['pid'] = proc.pid
                save_status(status)

            # Monitor until completion
            while proc.poll() is None:
                time.sleep(15)
                update_overall_status()

            ret = proc.returncode
            if ret != 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Fold {target_fold} exited with code {ret}! Check {log_path}")
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: Fold {target_fold} finished successfully!")

        update_overall_status()
        time.sleep(5)


def run_monitor():
    """Supervisor loop that checks progress and runs final evaluation when all folds complete."""
    print("Starting supervisor monitor...")
    while True:
        status = update_overall_status()
        completed = get_completed_folds()

        # Check if all 10 folds are done
        if len(completed) >= 10:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALL 10 FOLDS COMPLETED!")
            # Run evaluation
            eval_log = os.path.join(LOGS_DIR, 'eval_cab2_kfold_final.log')
            print(f"Running aggregate evaluation: python kfold_patch/evaluate_kfold.py --models cab2 > {eval_log}")
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '1'
            cmd = [sys.executable, os.path.join(_PROJECT_ROOT, 'kfold_patch/evaluate_kfold.py'), '--models', 'cab2']
            with open(eval_log, 'w') as ef:
                subprocess.run(cmd, cwd=_PROJECT_ROOT, env=env, stdout=ef, stderr=subprocess.STDOUT)
            print("Evaluation completed!")
            break

        time.sleep(30)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if mode == 'worker':
        gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        order = sys.argv[3] if len(sys.argv) > 3 else ('desc' if gpu == 2 else 'asc')
        run_worker_gpu(gpu, order=order)
    elif mode == 'monitor':
        run_monitor()
    elif mode == 'status':
        st = update_overall_status()
        with open(SUMMARY_FILE, 'r') as f:
            print(f.read())
    else:
        print(f"Unknown mode: {mode}. Use 'worker <gpu_id> [asc|desc]', 'monitor', or 'status'.")

