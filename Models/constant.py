
import glob
import os
from pathlib import Path


BASE_DIR    = Path(__file__).resolve().parent          
PROJECT_DIR = BASE_DIR.parent                           
DATASET_DIR = PROJECT_DIR / "Dataset construction" / "FINAL_DATASET"
LOG_DIR    = BASE_DIR / "LOGS"

# XGBoost thread count. Do NOT use -1 here: XGBoost sizes its OpenMP pool from
# the logical CPU count, and on WSL2 (22 logical / 11 physical cores) a pool
# spanning every logical CPU collapses under spin-wait barrier contention
# across oversubscribed SMT siblings. A 100-tree fit on 431 rows costs ~16 s
# with 22 threads versus ~0.02 s with 1. Capping at the physical core count
# keeps fits in the millisecond range.
def physical_cores() -> int:
    cores = {
        (open(p.replace("core_id", "physical_package_id")).read().strip(),
         open(p).read().strip())
        for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/core_id")
    }
    return len(cores) if cores else max(1, (os.cpu_count() or 2) // 2)


N_JOBS = max(1, min(physical_cores(), os.cpu_count() or 1))

STRING_MATCHING_DATASET = 'matching_horizon_7'
BERT_DATASET = 'BERT_horizon_3'


BASELINE_APPROACHES = {
    'same_as_last',
    'moving_average',
    'arima',
    'persistence_surge',
    'logistic_calls',
    'majority_class',
}

