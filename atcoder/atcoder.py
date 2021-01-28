from database import Database, LoadDatabase
from numba import njit, vectorize
import matplotlib.pyplot as plt
from atcoder import *
import numpy as np
import pickle
import time
import bz2
import os

os.environ['NUMBA_DISABLE_INTEL_SVML']  = '1'
CENTER = 1200
RATEDBOUND = np.inf


def prepare_data(db):
    CALCS_FILE = "calcs.pickle.bz2"    
    # if calculated and save before, load it from file
    if os.path.exists(CALCS_FILE):
        with bz2.BZ2File(CALCS_FILE, "r") as infile:
            print("Starting loading calcs file ...")
            ret = pickle.load(infile)
            print("File read.")
    else:
        print("Starting calcs ...")
        # load database
        db = LoadDatabase()

        # collect all handles in all standings
        all_handles = set()
        for standings in db.standings.values():
            for handle in standings.index:
                all_handles.add(handle)

        # create to way mappings (id, handle)
        handle_to_id = {handle: i for i, handle in enumerate(all_handles)}
        id_to_handle = {i: handle for handle, i in handle_to_id.items()}

        # sort standings by startTime
        sorted_standings = [(k, v) for k, v in sorted(db.standings.items(), key=lambda x: db.contests.loc[x[0]].startTime)]

        # merge handles, ranks and standings length into flat array
        handle_ids_merged = []
        ranks_merged = []
        standings_lengths_merged = []

        for c_id, standings in sorted_standings:
            standings = standings.sort_values("rank")
            for handle in standings.index:
                handle_ids_merged.append(handle_to_id[handle])
                ranks_merged.append(standings["rank"][handle])
            standings_lengths_merged.append(len(standings))

        # convert them to numpy array
        handle_ids = np.array(handle_ids_merged, dtype=np.int32)
        ranks = np.array(ranks_merged, dtype=np.int32)
        standings_lens = np.array(standings_lengths_merged, dtype=np.int32)
        user_contest_cnt = np.bincount(handle_ids)

        with bz2.BZ2File(CALCS_FILE, "w") as outfile:
            ret = (handle_to_id, id_to_handle, sorted_standings, handle_ids, ranks, standings_lens, user_contest_cnt)
            pickle.dump(ret, outfile)

        print("Calcs ended.")
        
    return ret


def get_first_K_contests(K, handle_ids, ranks, standings_lens, user_contest_cnt):
    if K == -1:
        return handle_ids, ranks, standings_lens, user_contest_cnt
    K_standings_len = np.sum(standings_lens[:K])
    K_handle_ids = handle_ids[:K_standings_len]
    K_ranks = ranks[:K_standings_len]
    K_standings_lens = standings_lens[:K]
    K_user_contest_cnt = np.bincount(K_handle_ids)
    return K_handle_ids, K_ranks, K_standings_lens, K_user_contest_cnt


@vectorize
def powersum(q, n):
    return q * (1 - q**n) / (1 - q)


@vectorize
def g(x):
    return np.power(2, x / 800)


@vectorize
def ginv(y):
    return 800 * np.log2(y)


@vectorize
def F(n):
    return np.sqrt(powersum(0.81, n)) / powersum(0.9, n)


@vectorize
def f(n):
    Finf = np.sqrt(0.81 / (1.0 - 0.81)) / (0.9 / (1.0 - 0.9))
    return (F(n) - Finf) / (F(1) - Finf) * CENTER


@njit(fastmath=True)
def atcoder_calculate(handle_ids, ranks, standings_lens, user_contest_cnt, verbose=True):
    user_cnt = len(user_contest_cnt)
    standings_cnt = len(standings_lens)
    history_cnt = len(handle_ids)
    
    # AtCoder stuff
    ranks = ranks.copy().astype(np.float64)
    nums = np.zeros(user_cnt, dtype=np.float64)
    dens = np.zeros(user_cnt, dtype=np.float64)
    aperfs = np.full(user_cnt, CENTER, dtype=np.float64)
    perfs = np.empty(history_cnt, dtype=np.float64)
    ratings = np.zeros(history_cnt, dtype=np.float64)
    offsets = np.cumsum(user_contest_cnt) - user_contest_cnt
    local_offsets = np.zeros(user_cnt, dtype=np.int32)
    current_ranks = np.empty_like(ranks, dtype=np.float64)
    
    # parallel binsearch stuff
    ls = np.empty(np.max(standings_lens), dtype=np.float64)
    rs = np.empty(np.max(standings_lens), dtype=np.float64)
    cnts = np.empty(np.max(standings_lens), dtype=np.int32)
    handles = np.empty(np.max(standings_lens), dtype=np.int32)
    ls_next = np.empty_like(ls)
    rs_next = np.empty_like(rs)
    cnts_next = np.empty_like(cnts)
    handle_to_rank = np.empty(user_cnt, dtype=np.int32)
    
    standings_offset = 0
    standings_left = len(standings_lens)
    
    for i in range(standings_cnt):
        if verbose:
            print("Standings left:", standings_left)
        standings_left -= 1
        standings_len = standings_lens[i]
        
        # fix ranks
        j = 0
        while j < standings_len:
            rank = ranks[standings_offset + j]
            k = j
            while k + 1 < standings_len and ranks[standings_offset + k + 1] == rank:
                k += 1
            ranks[j:k + 1] = (2 * rank + k - j) / 2
            j = k + 1
            
        # create handle -> rank mapping given current standings
        slice_l, slice_r = standings_offset, standings_offset + standings_len
        handle_to_rank[handle_ids[slice_l:slice_r]] = ranks[slice_l:slice_r]
        
        # prepare to parallel binsearch
        ls[0], rs[0] = 0, 5000
        cnts[0] = standings_len
        handles[:standings_len] = handle_ids[slice_l:slice_r]
        segs, segs_next = 1, 0
        handles_offset = 0
        max_iters = 80
        
        # do parallel binsearch
        for j in range(max_iters):
            updated = False
            
            for k in range(segs):
                l, r = ls[k], rs[k]
                cnt = cnts[k]
                
                if (r - l) <= 1e-1:
                    ls_next[segs_next] = l
                    rs_next[segs_next] = r
                    cnts_next[segs_next] = cnt
                    handles_offset += cnt
                    segs_next += 1
                else:
                    updated = True
                    m = (l + r) / 2
                    
                    val = 0.0
                    for t in range(standings_len):
                        handle_id = handle_ids[standings_offset + t]
                        aperf = aperfs[handle_id]
                        val += 1 / (1 + np.power(6, (m - aperf) / 400))
                        
                    lit, rit = handles_offset, handles_offset + cnt - 1
                    lefts, rights = 0, 0
                    while lit < rit:
                        lhandle_id = handles[lit]
                        lrank = handle_to_rank[lhandle_id]
                        if val <= lrank - 0.5:
                            lit += 1
                            lefts += 1
                            continue
                            
                        rhandle_id = handles[rit]
                        rrank = handle_to_rank[rhandle_id]
                        if val > rrank - 0.5:
                            rit -= 1
                            rights += 1
                            continue
                            
                        lefts += 1
                        rights += 1
                        handles[lit], handles[rit] = handles[rit], handles[lit]
                        lit += 1
                        rit -= 1
                        
                    if lit == rit:
                        handle_id = handles[lit]
                        rank = handle_to_rank[handle_id]
                        if val <= rank - 0.5:
                            lefts += 1
                        else:
                            rights += 1
                            
                    if lefts > 0:
                        ls_next[segs_next] = l
                        rs_next[segs_next] = m
                        cnts_next[segs_next] = lefts
                        segs_next += 1
                        
                    if rights > 0:
                        ls_next[segs_next] = m
                        rs_next[segs_next] = r
                        cnts_next[segs_next] = rights
                        segs_next += 1
                        
                    handles_offset += cnt
                    
            if not updated:
                break
                
            segs = segs_next
            ls[:segs] = ls_next[:segs]
            rs[:segs] = rs_next[:segs]
            cnts[:segs] = cnts_next[:segs]
            segs_next = 0
            handles_offset = 0
            
        # calculate perfs, ratings, ... after parallel binsearch
        handles_offset = 0
        for j in range(segs):
            perf_base = ls[j]
            cnt = cnts[j]
            
            for k in range(cnt):
                handle_id = handles[handles_offset + k]
                offset = offsets[handle_id]
                local_offset = local_offsets[handle_id]
                
                if local_offset == 0:
                    perf = (perf_base - CENTER) * 1.5 + CENTER
                    ratings[offset + local_offset] = CENTER
                else:
                    perf = perf_base
                    den = dens[handle_id]
                    last_sum = g(ratings[offset + local_offset - 1]) * den
                    rperf = min(perfs[offset + local_offset - 1], RATEDBOUND + 400)
                    ratings[offset + local_offset] = ginv((0.9 * (last_sum + g(rperf))) / (0.9 * (1 + den)))
                
                perfs[offset + local_offset] = perf
                nums[handle_id] = 0.9 * (perf + nums[handle_id])
                dens[handle_id] = 0.9 * (1 + dens[handle_id])
                aperfs[handle_id] = nums[handle_id] / dens[handle_id]
                
            handles_offset += cnt
            
        # move user ratings to one place
        for j in range(standings_len):
            handle_id = handle_ids[standings_offset + j]
            offset = offsets[handle_id]
            local_offset = local_offsets[handle_id]
            current_ranks[standings_offset + j] = ratings[offset + local_offset]
            local_offsets[handle_id] += 1
            
        standings_offset += standings_len
    
    return nums, dens, aperfs, perfs, ratings, offsets, local_offsets, current_ranks


@njit(fastmath=True)
def calculate_errors(err_fun, standings_lens, ranks, current_ranks, Is, verbose=True):
    standings_cnt = len(standings_lens)
    current_ranks = current_ranks.copy()
    errors = np.empty(standings_cnt, dtype=np.float64)
    
    standings_offset = 0
    standings_left = standings_cnt
    
    for i in range(standings_cnt):
        if verbose:
            print("Standings left:", standings_left)
        standings_left -= 1
        standings_len = standings_lens[i]
        
        # replace ratings with ranks
        j = 0
        while j < standings_len:
            current_rank = current_ranks[Is[standings_offset + j]]
            k = j
            while k + 1 < standings_len and current_ranks[Is[standings_offset + k + 1]] == current_rank:
                k += 1
            first = j + 1
            last = k + 1
            current_ranks[Is[standings_offset + j:standings_offset + k + 1]] = (first + last) / 2
            j = k + 1
        
        # calculate errors
        err = 0
        for j in range(standings_len):
            real_rank = ranks[standings_offset + j]
            expected_rank = current_ranks[standings_offset + j]
            err += err_fun(real_rank, expected_rank)
        errors[i] = err / standings_len
        
        standings_offset += standings_len
        
    return errors


# Additional return value of AtCoderRatingSystem, which has all calculations, meaningful variables (pretty specific stuff)
class Result:
    def __init__(self, consider, handle_to_id, id_to_handle, sorted_standings, handle_ids, ranks, standings_lens,
                 user_contest_cnt, nums, dens, aperfs, perfs, ratings, offsets, local_offsets, current_ranks,
                 Is, errors):
        self.consider = consider
        self.handle_to_id = handle_to_id
        self.id_to_handle = id_to_handle
        self.sorted_standings = sorted_standings
        self.handle_ids = handle_ids
        self.ranks = ranks
        self.standings_lens = standings_lens
        self.user_contest_cnt = user_contest_cnt
        self.nums = nums
        self.dens = dens
        self.aperfs = aperfs
        self.perfs = perfs
        self.ratings = ratings
        self.offsets = offsets
        self.local_offsets = local_offsets
        self.current_ranks = current_ranks
        self.Is = Is
        self.errors = errors
        
    def get_cf_ratings(self, handle):
        ratings = []
        if self.consider == -1:
            trimmed_standings = self.sorted_standings
        else:
            trimmed_standings = self.sorted_standings[:self.consider]
        for contest_id, standings in trimmed_standings:
            if handle in standings.index:
                ratings.append(standings.loc[handle]["oldRating"])
        return ratings
    
    def get_random_user(self, threshold=10):
        all_ids = np.arange(len(self.user_contest_cnt))
        mask = self.user_contest_cnt >= threshold
        handle_id = np.random.choice(all_ids[mask])
        return self.id_to_handle[handle_id]
    
    def plot_user(self, handle, verbose=False):
        handle_id = self.handle_to_id[handle]
        contest_cnt = self.user_contest_cnt[handle_id]
        user_offset = self.offsets[handle_id]
        print(contest_cnt, self.local_offsets[handle_id])
        assert contest_cnt == self.local_offsets[handle_id]
        
        perfs = self.perfs[user_offset:user_offset+contest_cnt]
        atcoder_ratings = self.ratings[user_offset:user_offset+contest_cnt]
        cf_ratings = self.get_cf_ratings(handle)
        
        assert contest_cnt == len(cf_ratings)
        print("number of contests", contest_cnt)
        
        if verbose:
            print("perfs", perfs)
            print("aperf", self.aperfs[handle_id])
            print("num", self.nums[handle_id])
            print("den", self.dens[handle_id])
            
        xs = np.arange(contest_cnt)
        plt.figure(figsize=(15, 8))
        plt.plot(xs, atcoder_ratings, label="AtCoder")
        plt.plot(xs, cf_ratings, label="CodeForces")
#         plt.plot(xs, perfs, label="AtCoder Perfs")
        plt.title(handle)
        plt.legend()
        plt.show()


# - return tuple (errors, results), where
#       results: Result class described above
#       errors: dictionary of: error_function_name -> (dictionary of: contest id -> error calculated with that function)
# - consider only `consider` first contests, if consider == -1, all contests are taken
# - `err_fun` parameter is one function or list of functions to calculate error with
def AtCoderRatingSystem(db, err_fun=None, consider=50, verbose=False, **kwargs):
    global handle_to_id, id_to_handle, sorted_standings, handle_ids, ranks, standings_lens, user_contest_cnt
    # get data in familiar form
    if not "handle_to_id" in globals():
        handle_to_id, id_to_handle, sorted_standings, handle_ids, ranks, standings_lens, user_contest_cnt = \
            prepare_data(db)
        
    # convert err_fun to list of jitted err funs
    try:
        iter(err_fun)
    except:
        err_fun = [err_fun]
    err_fun = list(map(njit(fastmath=True), err_fun))
    
    # compile (jit)
    compile_handle_ids, compile_ranks, compile_standings_lens, compile_user_contest_cnt = \
        get_first_K_contests(5,  handle_ids, ranks, standings_lens, user_contest_cnt)
    atcoder_calculate(compile_handle_ids, compile_ranks, compile_standings_lens, compile_user_contest_cnt,
                      verbose=False)

    # main calculations
    K_handle_ids, K_ranks, K_standings_lens, K_user_contest_cnt = \
        get_first_K_contests(consider, handle_ids, ranks, standings_lens, user_contest_cnt)
    t = time.time()
    nums, dens, aperfs, perfs, ratings, offsets, local_offsets, current_ranks = \
        atcoder_calculate(K_handle_ids, K_ranks, K_standings_lens, K_user_contest_cnt, verbose=verbose)
    delta = time.time() - t
    print("Calculated in %02dm %02.2fs" % (delta // 60, delta % 60))
    
    # some assertions haven't killed anybody, yet
    for i in range(len(K_user_contest_cnt)):
        assert K_user_contest_cnt[i] == local_offsets[i]
        
    # sort ratings in all contests to calculate error rate (argsort forbidden in numba ??)
    Is = np.empty_like(current_ranks, dtype=np.int32)
    standings_offset = 0
    standings_cnt = len(K_standings_lens)

    for i in range(standings_cnt):
        standings_len = K_standings_lens[i]
        slice_l, slice_r = standings_offset, standings_offset + standings_len
        Is[slice_l:slice_r] = standings_offset + np.argsort(current_ranks[slice_l:slice_r])
        standings_offset += standings_len
    
    # errors
    errors_dict = {}
    trimmed_sorted_standings = sorted_standings[:consider] if consider != -1 else sorted_standings
    for err_f in err_fun:
        errors = calculate_errors(err_f, K_standings_lens, ranks, current_ranks, Is, verbose=False)
        current_errors = {}
        for i, (contest_id, _) in enumerate(trimmed_sorted_standings):
            current_errors[contest_id] = errors[i]
        errors_dict[err_f.__name__] = current_errors
    
    return errors_dict, Result(consider, handle_to_id, id_to_handle, sorted_standings, K_handle_ids,
                               K_ranks, K_standings_lens, K_user_contest_cnt, nums, dens, aperfs, perfs,
                               ratings, offsets, local_offsets, current_ranks, Is, errors)


def sqrt_err(x, y):
    return np.sqrt(np.abs(x - y))

def linear_err(x, y):
    return np.abs(x - y)

def squared_err(x, y):
    return np.power(x - y, 2)


if __name__ == "__main__":
    print("Loading database ...")
    db = LoadDatabase()
    print("Loaded database. Starting calculations ...")
    errors, result = AtCoderRatingSystem(db,
        err_fun=[linear_err, sqrt_err, squared_err],
        consider=50,
        verbose=True)
    print("Calculations ended. Saving results ...")
    with open("results.pickle", "wb") as outfile:
        obj = (errors, result)
        pickle.dump(obj, outfile)
    print("Results saved.")