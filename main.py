import networkit as nk
import sys
import os
import math
import time
import pandas as pd
import copy

N_RUN = 5
DIR="./"
CACHE_DIR = './cache'
START_K = 3
END_K = 6
data_files = [
    "twitter_combined.txt",
    "com-amazon.ungraph.txt",
    "cit-patent.txt",
    "co-papers-citeseer.txt",
    "co-papers-dblp.txt",
    "NotreDame_www.txt",
]

# Function to read graph from the source            
def read_graph(fin):
    n = int(fin.readline())
    G = nk.Graph(n)
    
    # print("Number of nodes: ", n)    
    while True:
        try:
            line = fin.readline()
        except:
            break

        line = line.split()
        if len(line) == 0:
            break

        x = int(line[0][:-1])
        arr = [int(y) for y in line[1:]]        
        for y in arr:
            G.addEdge(x, y, addMissing = True)

    return G, n

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def inversion_count(p, B, aux, l, r):
    if l == r:
        return 0
    # if r - l < 50:
    #     cnt = 0
    #     for i in range(l+1, r+1):
    #         for j in range(l, i):
    #             if B[p[j]] > B[p[i]]:
    #                 cnt += 1
    #     return cnt
    mid = (l + r) // 2
    cnt = inversion_count(p, aux, B, l, mid)
    cnt += inversion_count(p, aux, B, mid+1, r)
    i, j, k = l, mid+1, l
    # eprint(l, mid, r)
    while i <= mid and j <= r:
        if aux[p[i]] >= aux[p[j]]:
            cnt += mid - i + 1
            B[p[k]] = aux[p[j]]
            j += 1
        else:
            B[p[k]] = aux[p[i]]
            i += 1
        k += 1
    while j <= r:
        B[p[k]] = aux[p[j]]
        j += 1
        k += 1
    while i <= mid:
        B[p[k]] = aux[p[i]]
        i += 1
        k += 1
    return cnt

def inversion(A, B):
    n = len(A)
    p = [i for i in range(n)]
    p = sorted(p, key = lambda x: A[x])
    return inversion_count(p, copy.copy(B), copy.copy(B), 0, n - 1)

def inversion_pct(A, B):
    n = len(A)
    m = inversion(A, B)
    return m*100/((n-1)*(n-2))
    
def printsorted(n, bc):
    p = [i for i in range(n)]
    p = sorted(p, key = lambda x: bc[x], reverse = True)
    for x in p:
        print(x+1, round(bc[x],2))
    

def similarity(A, B, topk):
    n = len(A)
    pa = [i for i in range(n)]
    pb = [i for i in range(n)]
    pa = sorted(pa, key = lambda x : A[x], reverse = True)
    pb = sorted(pb, key = lambda x : B[x], reverse = True)
    A = set([i for i in pa[:topk]])
    B = set([i for i in pb[:topk]])
    return len(A & B) / topk * 100

def similarity2(A, B, topk):
    n = len(A)
    pa = [i for i in range(n)]
    pb = [i for i in range(n)]
    pa = sorted(pa, key = lambda x : A[x], reverse = True)
    pb = sorted(pb, key = lambda x : B[x], reverse = True)
    A = set([i for i in pa[:2*topk]])
    B = set([i for i in pb[:topk]])
    return len(A & B)


def cache_data(fname, bc, time):
    with open(fname, 'w') as fout:
        fout.write(str(time) + '\n')
        for v in bc:
            fout.write(str(v) + ' ')
        fout.write('\n')

def read_cache(fname):
    bc = []
    with open(fname) as fin:
        bc_time = float(fin.readline())
        for val in fin.readline().split():
            bc.append(float(val))
    return bc_time, bc

def rank_nodes(bc, n):
    p = [i for i in range(n)]
    p = sorted(p, key = lambda x: bc[x], reverse = True)
    return p

def euclidean_distance(a, b, an = True, bn = True):
    a = copy.copy(a)
    b = copy.copy(b)
    d = 0
    n = len(a)
    if an:
        for i in range(n):
            a[i] /= (n-2)*(n-1)
    if bn:
        for i in range(n):
            b[i] /= (n-2)*(n-1)
    for i in range(len(a)):
        d += (a[i] - b[i])**2

    return math.sqrt(d)
    
if __name__ == '__main__':
    sys.setrecursionlimit(int(1e9))
    nk.setNumberOfThreads(1)
    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)


    for di, fname in enumerate(data_files):
        with open(DIR + "/" + fname, 'r') as fin:
            G, n = read_graph(fin)

        bc = [0] * n
        bc_time = 1
        cache_name = CACHE_DIR + '/' + 'bc_' + fname
        if di < 2:
            if os.path.exists(cache_name):
                bc_time, bc = read_cache(cache_name)
            else:
                start_time = time.perf_counter()
                bc = nk.centrality.Betweenness(G)
                bc.run()
                end_time = time.perf_counter()
                bc_time = end_time - start_time
                bc = bc.scores()
                cache_data(cache_name, bc, bc_time)

        # # bc_rank = rank_nodes(bc, n)
        topk = int(math.sqrt(n))

        ebc_avg_time = 0.0
        abc_avg_time = 0.0
        bc_ebc_avg = [0.0] * 3
        bc_abc_avg = [0.0] * 3
        avg_ebc = [0.0] * n
        avg_abc = [0.0] * n
        kbc_ebc_avg = [0.0] * 3
        kbc_abc_avg = [0.0] * 3
        bc_ebc_eu = 0
        bc_ebc_inv = 0
        bc_abc_eu = 0
        bc_abc_inv = 0


        for rep in range(N_RUN):
            cache_name = CACHE_DIR + '/' + 'ebc_' +  str(rep) + '_' + fname
            eprint(rep, "ebc")
            if os.path.exists(cache_name):
                ebc_time, ebc = read_cache(cache_name)
            else:
                start_time = time.perf_counter()
                ebc = nk.centrality.EstimateBetweenness(G, int(math.log2(n)**3))
                ebc.run()
                end_time = time.perf_counter()
                ebc_time = end_time - start_time
                ebc = ebc.scores()
                cache_data(cache_name, ebc, ebc_time)

            ebc_avg_time += ebc_time
            for i, v in enumerate(ebc):
                avg_ebc[i] += v
            bc_ebc_avg[0] += similarity2(bc, ebc, 1)
            bc_ebc_avg[1] += similarity2(bc, ebc, 10)
            bc_ebc_avg[2] += similarity2(bc, ebc, topk)
            bc_ebc_eu += euclidean_distance(bc, ebc)
            bc_ebc_inv += inversion_pct(bc, ebc)
            
            cache_name = CACHE_DIR + '/' + 'abc_' + str(rep) + '_' + fname
            eprint(rep, "abc")
            if os.path.exists(cache_name):
                abc_time, abc = read_cache(cache_name)
            else:
                start_time = time.perf_counter()
                abc = nk.centrality.ApproxBetweenness(G, delta = 0.1,  epsilon = 0.005)
                abc.run()
                end_time = time.perf_counter()
                abc_time = end_time - start_time
                abc = abc.scores()
                cache_data(cache_name, abc, abc_time)
                    
            abc_avg_time += abc_time
            for i, v in enumerate(abc):
                avg_abc[i] += v
            bc_abc_avg[0] += similarity2(bc, abc, 1)
            bc_abc_avg[1] += similarity2(bc, abc, 10)
            bc_abc_avg[2] += similarity2(bc, abc, topk)
            bc_abc_eu += euclidean_distance(bc, abc, True, False)
            bc_abc_inv += inversion_pct(bc, abc)

        for i in range(n):
            avg_ebc[i] /= N_RUN
            avg_abc[i] /= N_RUN
        ebc_avg_time /= N_RUN
        abc_avg_time /= N_RUN
        bc_ebc_eu /= N_RUN
        bc_ebc_inv /= N_RUN
        bc_abc_eu /= N_RUN
        bc_abc_inv /= N_RUN
        for i in range(3):
            bc_ebc_avg[i] /= N_RUN
            bc_abc_avg[i] /= N_RUN

        
        x = []
        for k in range(START_K, END_K):
            eprint('------------------------------------- ' + str(k) + ' --------------------------------------------------')
            x.append(k)
            cache_name = CACHE_DIR + '/' + 'kbc_' + str(k) + '_' + fname
            if os.path.exists(cache_name):
                kbc_time, kbc = read_cache(cache_name)
            else:
                start_time = time.perf_counter()
                kbc = nk.centrality.Betweenness(G, _K = k)
                kbc.run()
                end_time = time.perf_counter()
                kbc_time = end_time - start_time
                kbc = kbc.scores()
                cache_data(cache_name, kbc, kbc_time)

            # kbc_rank = rank_nodes(kbc, n)
            
            kbc_ebc_avg[0] += similarity2(kbc, avg_ebc, 1)
            kbc_ebc_avg[1] += similarity2(kbc, avg_ebc, 10)
            kbc_ebc_avg[2] += similarity2(kbc, avg_ebc, topk)
            kbc_abc_avg[0] += similarity2(kbc, avg_abc, 1)
            kbc_abc_avg[1] += similarity2(kbc, avg_abc, 10)
            kbc_abc_avg[2] += similarity2(kbc, avg_abc, topk)
            bc_kbc_eu = euclidean_distance(bc, kbc)
            bc_kbc_inv = inversion_pct(bc, kbc)
            kbc_ebc_eu = euclidean_distance(kbc, avg_ebc)
            kbc_ebc_inv = inversion_pct(kbc, avg_ebc)
            kbc_abc_eu = euclidean_distance(kbc, avg_abc, True, False)
            kbc_abc_inv = inversion_pct(kbc, avg_abc)


            ekbc_avg_time = 0.0
            bc_ekbc_avg = [0.0] * 3
            kbc_ekbc_avg = [0.0] * 3
            avg_ekbc = [0.0] * n
            akbc_avg_time = 0.0
            bc_akbc_avg = [0.0] * 3
            avg_akbc = [0.0] * n
            kbc_akbc_avg = [0.0] * 3
            bc_ekbc_eu = 0
            bc_ekbc_inv = 0
            bc_akbc_eu = 0
            bc_akbc_inv = 0
            kbc_ekbc_eu = 0
            kbc_ekbc_inv = 0
            kbc_akbc_eu = 0
            kbc_akbc_inv = 0

            
            for rep in range(N_RUN):
                cache_name = CACHE_DIR + '/' + 'ekbc_' + str(k) + '_' + str(rep) + '_' + fname
                eprint(rep, "ekbc")
                if os.path.exists(cache_name):
                    ekbc_time, ekbc = read_cache(cache_name)
                else:
                    start_time = time.perf_counter()
                    ekbc = nk.centrality.EstimateBetweenness(G, int(math.log2(n)**3), _K = k)
                    ekbc.run()
                    end_time = time.perf_counter()
                    ekbc_time = end_time - start_time
                    ekbc = ekbc.scores()
                    cache_data(cache_name, ekbc, ekbc_time)

                
                ekbc_avg_time += ekbc_time
                
                for i, v in enumerate(ekbc):
                    avg_ekbc[i] += v
                bc_ekbc_avg[0] += similarity2(bc, ekbc, 1)
                bc_ekbc_avg[1] += similarity2(bc, ekbc, 10)
                bc_ekbc_avg[2] += similarity2(bc, ekbc, topk)
                bc_ekbc_eu += euclidean_distance(bc, ekbc)
                bc_ekbc_inv += inversion_pct(bc, ekbc)
                kbc_ekbc_avg[0] += similarity2(kbc, ekbc, 1)
                kbc_ekbc_avg[1] += similarity2(kbc, ekbc, 10)
                kbc_ekbc_avg[2] += similarity2(kbc, ekbc, topk)
                kbc_ekbc_eu += euclidean_distance(kbc, ekbc)
                kbc_ekbc_inv += inversion_pct(kbc, ekbc)


                cache_name = CACHE_DIR + '/' + 'akbc_' + str(k) + '_' + str(rep) + '_' + fname
                eprint(rep, "akbc")
                if os.path.exists(cache_name):
                    akbc_time, akbc = read_cache(cache_name)
                else:
                    start_time = time.perf_counter()
                    akbc = nk.centrality.ApproxBetweenness(G, delta = 0.1,  epsilon = 0.005, _K = k)
                    akbc.run()
                    end_time = time.perf_counter()
                    akbc_time = end_time - start_time
                    akbc = akbc.scores()
                    cache_data(cache_name, akbc, akbc_time)
                    
                akbc_avg_time += akbc_time
                for i, v in enumerate(akbc):
                    avg_akbc[i] += v
                bc_akbc_avg[0] += similarity2(bc, akbc, 1)
                bc_akbc_avg[1] += similarity2(bc, akbc, 10)
                bc_akbc_avg[2] += similarity2(bc, akbc, topk)
                bc_akbc_eu += euclidean_distance(bc, akbc, True, False)
                bc_akbc_inv += inversion_pct(bc, akbc)
                kbc_akbc_avg[0] += similarity2(kbc, akbc, 1)
                kbc_akbc_avg[1] += similarity2(kbc, akbc, 10)
                kbc_akbc_avg[2] += similarity2(kbc, akbc, topk)
                kbc_akbc_eu += euclidean_distance(kbc, akbc, True, False)
                kbc_akbc_inv += inversion_pct(kbc, akbc)

                
            for i in range(n):
                avg_ekbc[i] /= N_RUN
                avg_akbc[i] /= N_RUN

            ekbc_avg_time /= N_RUN
            akbc_avg_time /= N_RUN
            bc_ekbc_eu /= N_RUN
            bc_ekbc_inv /= N_RUN
            kbc_ekbc_eu /= N_RUN
            kbc_ekbc_inv /= N_RUN
            bc_akbc_eu /= N_RUN
            bc_akbc_inv /= N_RUN
            kbc_akbc_eu /= N_RUN
            kbc_akbc_inv /= N_RUN
            for i in range(3):
                bc_ekbc_avg[i] /= N_RUN
                bc_akbc_avg[i] /= N_RUN
                kbc_ekbc_avg[i] /= N_RUN
                kbc_akbc_avg[i] /= N_RUN

            

            # ekbc_rank = rank_nodes(avg_ekbc, n)
            # akbc_rank = rank_nodes(avg_akbc, n)

            topk_sz = [1, 10, topk]

            df_ekbc = []
            df_akbc = []
            for i in range(n):
                df_ekbc.append((bc[i], kbc[i], avg_ekbc[i], avg_ebc[i]))
                df_akbc.append((bc[i], kbc[i], avg_akbc[i], avg_abc[i]))

            df_ekbc = pd.DataFrame(df_ekbc, columns = ['bc', 'kbc', 'ekbc', 'ebc'])
            df_akbc = pd.DataFrame(df_akbc, columns = ['bc', 'kbc', 'akbc', 'abc'])
            sp_cor_ekbc = df_ekbc.corr(method = 'spearman')
            sp_cor_akbc = df_akbc.corr(method = 'spearman')
            pr_cor_ekbc = df_ekbc.corr(method = 'pearson')
            pr_cor_akbc = df_akbc.corr(method = 'pearson')

            print(round(sp_cor_ekbc['bc']['kbc'], 2), end = ' ')
            print(round(pr_cor_ekbc['bc']['kbc'], 2), end = ' ')
            for kk in [1, 10, topk]:
                print(similarity2(bc, kbc, kk), '/', kk, sep='', end = ' ')
            print(bc_kbc_eu, bc_kbc_inv, bc_time, kbc_time, end = ' ')

            print(round(sp_cor_ekbc['bc']['ebc'], 2), end = ' ')
            print(round(pr_cor_ekbc['bc']['ebc'], 2), end = ' ')
            for i in range(3):
                print(bc_ebc_avg[i], '/', topk_sz[i], sep='', end = ' ')
            print(bc_ebc_eu, bc_ebc_inv, bc_time, ebc_avg_time, end = ' ')

            print(round(sp_cor_akbc['bc']['abc'], 2), end = ' ')
            print(round(pr_cor_akbc['bc']['abc'], 2), end = ' ')
            for i in range(3):
                print(bc_abc_avg[i], '/', topk_sz[i], sep='', end = ' ')
            print(bc_abc_eu, bc_abc_inv, bc_time, abc_avg_time, end = ' ')

            print(round(sp_cor_ekbc['bc']['ekbc'], 2), end = ' ')
            print(round(pr_cor_ekbc['bc']['ekbc'], 2), end = ' ')
            for i in range(3):
                print(bc_ekbc_avg[i], '/', topk_sz[i], sep='', end = ' ')
            print(bc_ekbc_eu, bc_ekbc_inv, bc_time, ekbc_avg_time, end = ' ')

            print(round(sp_cor_akbc['bc']['akbc'], 2), end = ' ')
            print(round(pr_cor_akbc['bc']['akbc'], 2), end = ' ')
            for i in range(3):
                print(bc_akbc_avg[i], '/', topk_sz[i], sep='', end = ' ')
            print(bc_akbc_eu, bc_akbc_inv, bc_time, akbc_avg_time, end = ' ')

            print(round(sp_cor_ekbc['kbc']['ebc'], 2), end = ' ')
            print(round(pr_cor_ekbc['kbc']['ebc'], 2), end = ' ')
            for i in range(3):
                print(kbc_ebc_avg[i], '/', topk_sz[i], sep='', end = ' ')
            print(kbc_ebc_eu, kbc_ebc_inv, kbc_time, ebc_avg_time, end = ' ')

            print(round(sp_cor_akbc['kbc']['abc'], 2), end = ' ')
            print(round(pr_cor_akbc['kbc']['abc'], 2), end = ' ')
            for i in range(3):
                print(kbc_abc_avg[i], '/', topk_sz[i], sep='', end = ' ')
            print(kbc_abc_eu, kbc_abc_inv, kbc_time, abc_avg_time, end = ' ')

            print(round(sp_cor_ekbc['kbc']['ekbc'], 2), end = ' ')
            print(round(pr_cor_ekbc['kbc']['ekbc'], 2), end = ' ')
            for i in range(3):
                print(kbc_ekbc_avg[i], '/', topk_sz[i], sep='', end = ' ')
            print(kbc_ekbc_eu, kbc_ekbc_inv, kbc_time, ekbc_avg_time, end = ' ')

            print(round(sp_cor_akbc['kbc']['akbc'], 2), end = ' ')
            print(round(pr_cor_akbc['kbc']['akbc'], 2), end = ' ')
            for i in range(3):
                print(kbc_akbc_avg[i], '/', topk_sz[i], sep='', end = ' ')
            print(kbc_akbc_eu, kbc_akbc_inv, kbc_time, akbc_avg_time, end = ' ')
            print()

