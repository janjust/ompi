/*
 * slowpatch.c - LD_PRELOAD that widens apply_patch()'s race window
 *
 * apply_patch() in opal/mca/patcher/base/patcher_base_patch.c does:
 *
 *   mprotect(addr, size, PROT_EXEC|PROT_READ|PROT_WRITE)   <- makes page writable
 *   memcpy(addr, patch_bytes, size)                         <- NON-ATOMIC write
 *   mprotect(addr, size, PROT_EXEC|PROT_READ)               <- removes write perm
 *
 * The memcpy is 13 bytes on x86_64 and 20 bytes on AArch64, taking ~3-5 ns.
 * Any thread executing mmap() during that window encounters partially-overwritten
 * machine code and crashes.
 *
 * This library:
 *   1. Intercepts mprotect(): tracks which pages are currently EXEC+WRITE
 *      (i.e. currently being patched).
 *   2. Intercepts memcpy(): when the destination is an EXEC+WRITE tracked page,
 *      replaces the atomic memcpy with a byte-by-byte loop with a configurable
 *      sleep between each byte, widening the window from ~5 ns to seconds.
 *
 * Build:
 *   gcc -shared -fPIC -O0 -o slowpatch.so slowpatch.c -ldl -lpthread
 *
 * Use:
 *   LD_PRELOAD=./slowpatch.so PATCH_SLEEP_US=100000 mpirun -np 1 ./race_mmap 16
 *
 * Environment:
 *   PATCH_SLEEP_US   microseconds to sleep between each byte (default: 100000 = 100ms)
 *                    With 13 bytes on x86_64: 13 * 100ms = 1.3s total window.
 *                    With 20 bytes on AArch64: 20 * 100ms = 2.0s total window.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/mman.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Real function pointers, initialised once via constructor            */
/* ------------------------------------------------------------------ */
static void *(*real_memcpy)(void *, const void *, size_t) = NULL;
static int   (*real_mprotect)(void *, size_t, int)        = NULL;

__attribute__((constructor))
static void init_real_syms(void)
{
    real_memcpy   = dlsym(RTLD_NEXT, "memcpy");
    real_mprotect = dlsym(RTLD_NEXT, "mprotect");
}

/* ------------------------------------------------------------------ */
/* Track pages that currently have EXEC+WRITE permission               */
/* (i.e. are mid-patch in apply_patch)                                 */
/* ------------------------------------------------------------------ */
#define MAX_EXEC_WRITE_PAGES 64
static uintptr_t exec_write_pages[MAX_EXEC_WRITE_PAGES];
static int       n_exec_write_pages = 0;
static pthread_mutex_t pages_mtx = PTHREAD_MUTEX_INITIALIZER;

static void page_mark(uintptr_t page)
{
    pthread_mutex_lock(&pages_mtx);
    if (n_exec_write_pages < MAX_EXEC_WRITE_PAGES)
        exec_write_pages[n_exec_write_pages++] = page;
    pthread_mutex_unlock(&pages_mtx);
}

static void page_unmark(uintptr_t page)
{
    pthread_mutex_lock(&pages_mtx);
    for (int i = 0; i < n_exec_write_pages; i++) {
        if (exec_write_pages[i] == page) {
            exec_write_pages[i] = exec_write_pages[--n_exec_write_pages];
            break;
        }
    }
    pthread_mutex_unlock(&pages_mtx);
}

static int page_is_exec_write(const void *addr)
{
    uintptr_t page = (uintptr_t)addr & ~(uintptr_t)4095;
    int found = 0;
    pthread_mutex_lock(&pages_mtx);
    for (int i = 0; i < n_exec_write_pages; i++) {
        if (exec_write_pages[i] == page) { found = 1; break; }
    }
    pthread_mutex_unlock(&pages_mtx);
    return found;
}

/* ------------------------------------------------------------------ */
/* mprotect interposer                                                  */
/* ------------------------------------------------------------------ */
int mprotect(void *addr, size_t len, int prot)
{
    int ret = real_mprotect(addr, len, prot);
    uintptr_t page = (uintptr_t)addr & ~(uintptr_t)4095;

    if ((prot & (PROT_EXEC | PROT_WRITE)) == (PROT_EXEC | PROT_WRITE)) {
        page_mark(page);
    } else {
        /* Write permission removed: patching window closed */
        page_unmark(page);
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/* memcpy interposer                                                    */
/* ------------------------------------------------------------------ */

/*
 * Guard against recursive interception: if this thread is already inside
 * our slow-path memcpy (e.g. fprintf calls memcpy internally), fall through
 * to the real memcpy immediately.
 */
static __thread int in_slow_memcpy = 0;

void *memcpy(void *dst, const void *src, size_t n)
{
    /*
     * Only intercept small writes to executable pages — apply_patch writes
     * 13 bytes (x86_64), 20 bytes (AArch64), 28 bytes (PPC64), never more.
     * This avoids slowing down unrelated memcpy calls.
     */
    if (!in_slow_memcpy && n > 0 && n <= 32 && page_is_exec_write(dst)) {
        in_slow_memcpy = 1;

        const char *env   = getenv("PATCH_SLEEP_US");
        useconds_t  delay = env ? (useconds_t)atoi(env) : 100000u;

        fprintf(stderr,
                "[slowpatch] intercepted %zu-byte write to exec page %p "
                "(sleeping %u us per byte = %.1f s total)\n",
                n, dst, (unsigned)delay, (double)n * delay / 1e6);
        fflush(stderr);

        volatile unsigned char *d = (volatile unsigned char *)dst;
        const  unsigned char   *s = (const  unsigned char   *)src;

        for (size_t i = 0; i < n; i++) {
            d[i] = s[i];
            /*
             * Sleep AFTER each byte so threads calling mmap() will execute
             * partially-overwritten instruction bytes:
             *   i==0:  only first byte written  (e.g. 0x49 REX prefix on x86_64)
             *   i==9:  mov r11,<addr> complete, jmp r11 not yet written
             *   i==12: full patch written (no longer a race)
             */
            usleep(delay);
        }

        fprintf(stderr, "[slowpatch] patch write complete (%zu bytes)\n", n);
        fflush(stderr);

        in_slow_memcpy = 0;
        return dst;
    }

    /* Guard against calling NULL during early library-constructor ordering */
    if (real_memcpy == NULL) {
        volatile char *d = (volatile char *)dst;
        const char    *s = (const char *)src;
        for (size_t i = 0; i < n; i++) d[i] = s[i];
        return dst;
    }
    return real_memcpy(dst, src, n);
}
