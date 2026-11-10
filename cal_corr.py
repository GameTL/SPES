import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import time

def cal_correlation(pre_arr, arr, lag):
    if lag > 0:
        pre = np.where(np.array(pre_arr[:-lag])>0, True, False)
        cur = np.where(np.array(arr[lag:])>0, True, False)
    else:
        pre = np.where(np.array(pre_arr)>0, True, False)
        cur = np.where(np.array(arr)>0, True, False)
    
    if sum(cur) > 0:
        return sum(pre & cur) / sum(cur), sum(pre & (~cur)) / sum(~cur)
    else:
        return 0, 0


def load_corr_inputs(corr_dir):
    func_arrcount = {}
    func_forget = {}
    with open(os.path.join(corr_dir, "to_cal_corr.txt")) as rf:
        for line in rf:
            func, forget, arrcount = line.strip().split('\t')
            func_forget[func] = int(forget)
            # int32 keeps the full minute series under 16GB. cal_correlation
            # already wraps inputs in np.array; the >0 predicate is unchanged.
            func_arrcount[func] = np.asarray(arrcount.split(','), dtype=np.int32)

    unknown_ownerapp_lst = {}
    with open(os.path.join(corr_dir, "unknown_ownerapp_lst.txt")) as rf:
        for line in rf:
            func, ownerapp_lst = line.strip().split('\t')
            unknown_ownerapp_lst[func] = ownerapp_lst.split(',')
    return func_arrcount, func_forget, unknown_ownerapp_lst


def run_shard(start, total, warm_th, func_arrcount, func_forget, unknown_ownerapp_lst, out_dir):
    """Same shard loop as the original CLI. Writes one file: {out_dir}/{start}.txt."""
    zero_th = 1 - warm_th
    pre_func_lst = defaultdict(list)
    keys = sorted(list(unknown_ownerapp_lst.keys()))
    shard = keys[start * total: min((start + 1) * total, len(keys))]

    c = 0
    st = time.time()
    with tqdm(total=len(shard) if len(shard) else total) as pbar:
        for func in shard:
            c += 1
            if c % 40 == 0:
                pbar.update(40)
            for pre_func in set(unknown_ownerapp_lst[func]):
                if pre_func == func:
                    continue
                forget = func_forget[pre_func]
                for lag in range(0, 11):
                    corr_rate, zero_inconsistency = cal_correlation(
                        func_arrcount[pre_func][forget * 1440:],
                        func_arrcount[func][forget * 1440:],
                        lag,
                    )
                    if corr_rate >= warm_th and zero_inconsistency <= zero_th:
                        pre_func_lst[func].append((pre_func, lag, corr_rate))
                        break
            if func in pre_func_lst:
                pre_func_lst[func] = sorted(pre_func_lst[func], key=lambda tri_tuple: tri_tuple[-1], reverse=True)

    print(len(pre_func_lst), round((time.time() - st) / 60, 4))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{start}.txt")
    with open(out_path, "w") as wf:
        for f, tri_tuple_lst in pre_func_lst.items():
            wf.write(f + ":" + '\t'.join("{},{},{:.4f}".format(a, b, c) for (a, b, c) in tri_tuple_lst) + '\n')
    return out_path


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--total", default=3704, type=int)  # 1/10 of the total
    parser.add_argument("--warm_th", default=0.7, type=float)
    parser.add_argument("--corr-dir", default="./cal_corr")
    parser.add_argument("--out-dir", default="./func_candi_tuple_lst")
    params = vars(parser.parse_args())
    func_arrcount, func_forget, unknown_ownerapp_lst = load_corr_inputs(params["corr_dir"])
    run_shard(
        params["start"],
        params["total"],
        params["warm_th"],
        func_arrcount,
        func_forget,
        unknown_ownerapp_lst,
        params["out_dir"],
    )
