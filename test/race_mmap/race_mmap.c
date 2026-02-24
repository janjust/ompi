/*
 * race_mmap.c - Reproducer for the apply_patch() binary-patching race condition
 *
 * The OPAL memory patcher (opal/mca/patcher/overwrite/patcher_overwrite_module.c)
 * binary-patches mmap/munmap/madvise by overwriting the function's entry bytes
 * with a non-atomic memcpy of 13 bytes (x86_64) or 20 bytes (AArch64).
 * This is triggered during MPI_Init when the UCX PML component opens and calls
 * opal_common_ucx_mca_register() -> mca_base_framework_open(memory) -> patcher_open().
 *
 * If any thread executes mmap() during the ~3-5ns memcpy window it will fetch
 * partially-overwritten machine code and crash.
 *
 * --- How to build and run ---
 *
 * 1. Narrow window (statistical, may need many runs):
 *      mpicc -O2 -o race_mmap race_mmap.c -lpthread
 *      mpirun -np 1 ./race_mmap [num_threads]    # default: 64
 *
 * 2. Wide window (virtually guaranteed, requires slowpatch.so):
 *      gcc -shared -fPIC -O0 -o slowpatch.so slowpatch.c -ldl -lpthread
 *      mpicc -O2 -o race_mmap race_mmap.c -lpthread
 *      LD_PRELOAD=./slowpatch.so mpirun -np 1 ./race_mmap 16
 *
 *      The LD_PRELOAD intercepts memcpy() and, when it detects a write to an
 *      executable page (i.e. apply_patch's patching memcpy), replaces it with
 *      a byte-by-byte loop with PATCH_SLEEP_US microseconds between bytes.
 *      Default: 100 000 us (100 ms) per byte => 1.3s window on x86_64.
 *
 * --- Why direct mmap() and not malloc() ---
 *
 * The patcher patches the mmap *symbol* entry bytes. malloc() uses per-thread
 * arena sbrk()/brk() for most small allocations and only calls mmap()
 * occasionally to grow an arena. Direct mmap()/munmap() calls bypass the
 * allocator entirely and go straight through the symbol being patched,
 * maximising the collision rate with the patching window.
 *
 * --- Expected outcome ---
 *
 * Without slowpatch.so: probable crash (SIGSEGV/SIGILL/SIGBUS) within a few
 *   seconds on AArch64 (20-byte patch, fixed-width instruction words).
 *   On x86_64 may take longer due to the narrower 13-byte window and variable-
 *   length instruction encoding allowing some partial states to be "valid".
 *
 * With slowpatch.so: near-certain crash within the first patched symbol.
 */

#define _GNU_SOURCE
#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdatomic.h>
#include <string.h>
#include <errno.h>

#define DEFAULT_NUM_WORKERS  64
#define MAP_SIZE             4096

static volatile int  g_stop  = 0;
static atomic_long   g_count = 0;
static atomic_long   g_fails = 0;

static void *worker(void *arg)
{
    (void)arg;
    long c = 0, f = 0;

    while (!g_stop) {
        /*
         * Direct mmap/munmap: every call goes through the symbol being
         * patched. MAP_ANONYMOUS + MAP_PRIVATE: no fd, no file backing,
         * pure virtual address allocation - the fastest path.
         */
        void *p = mmap(NULL, MAP_SIZE,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS,
                       -1, 0);
        if (p == MAP_FAILED) {
            f++;
            continue;
        }
        /* Touch the page so the kernel actually faults it in */
        ((volatile char *)p)[0] = 1;
        munmap(p, MAP_SIZE);
        c++;
    }

    atomic_fetch_add(&g_count, c);
    atomic_fetch_add(&g_fails, f);
    return NULL;
}

int main(int argc, char **argv)
{
    int nw = (argc > 1) ? atoi(argv[1]) : DEFAULT_NUM_WORKERS;
    if (nw < 1)   nw = 1;
    if (nw > 512) nw = 512;

    fprintf(stderr, "Spawning %d worker threads doing tight mmap/munmap loops\n", nw);
    fflush(stderr);

    pthread_t *tids = malloc((size_t)nw * sizeof(pthread_t));
    if (!tids) { perror("malloc"); return 1; }

    for (int i = 0; i < nw; i++) {
        if (pthread_create(&tids[i], NULL, worker, NULL) != 0) {
            perror("pthread_create");
            nw = i;
            break;
        }
    }

    /*
     * Give all workers time to reach their mmap loops and enter a steady
     * rate before we trigger the patching via MPI_Init_thread.
     */
    usleep(50000); /* 50 ms */

    fprintf(stderr, "Calling MPI_Init_thread (triggers apply_patch on mmap) ...\n");
    fflush(stderr);

    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed\n");
        g_stop = 1;
        for (int i = 0; i < nw; i++) pthread_join(tids[i], NULL);
        free(tids);
        return 1;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    fprintf(stderr, "[rank %d/%d] MPI_Init_thread returned OK, provided=%d\n",
            rank, size, provided);

    g_stop = 1;
    for (int i = 0; i < nw; i++) pthread_join(tids[i], NULL);

    fprintf(stderr, "[rank %d] mmap calls: %ld  mmap failures: %ld\n",
            rank, atomic_load(&g_count), atomic_load(&g_fails));

    free(tids);
    MPI_Finalize();
    fprintf(stderr, "Done.\n");
    return 0;
}
