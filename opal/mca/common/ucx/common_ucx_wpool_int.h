#ifndef COMMON_UCX_WPOOL_INT_H
#define COMMON_UCX_WPOOL_INT_H

#include "opal_config.h"
#include "common_ucx.h"
#include "common_ucx_wpool.h"

struct __tlocal_ctx_t {
    opal_common_ucx_ctx_t *gctx;
    opal_common_ucx_winfo_t *winfo;
    opal_atomic_int32_t refcnt;
};

struct __mem_info_t {
    opal_common_ucx_winfo_t *worker;
    ucp_rkey_h *rkeys;
};

// struct __tlocal_mem_t {
//     opal_common_ucx_wpmem_t *gmem;
//     _mem_info_t *mem;
//     opal_common_ucx_tlocal_fast_ptrs_t *mem_tls_ptr;
//     _tlocal_ctx_t *ctx_rec;
// };

typedef struct {
    opal_list_item_t super;
    opal_common_ucx_winfo_t *ptr;
} _winfo_list_item_t;
OBJ_CLASS_DECLARATION(_winfo_list_item_t);

typedef struct {
    opal_list_item_t super;
    opal_common_ucx_winfo_t *ptr;
} _ctx_record_list_item_t;
OBJ_CLASS_DECLARATION(_ctx_record_list_item_t);

static int _tlocal_ctx_connect(_tlocal_ctx_t *ctx, int target);
static int _tlocal_mem_create_rkey(_tlocal_mem_t *mem_rec, ucp_ep_h ep, int target);

/* Sorted declarations */

/* Internal Worker Information (winfo) management */
static opal_common_ucx_winfo_t *_winfo_create(opal_common_ucx_wpool_t *wpool);
static void _winfo_release(opal_common_ucx_winfo_t *winfo);
static void _winfo_reset(opal_common_ucx_winfo_t *winfo);

/* Internal Worker Pool (wpool) management */
static int _wpool_list_put(opal_common_ucx_wpool_t *wpool, opal_list_t *list,
                           opal_common_ucx_winfo_t *winfo);
static int _wpool_list_put(opal_common_ucx_wpool_t *wpool, opal_list_t *list,
                           opal_common_ucx_winfo_t *winfo);
static opal_common_ucx_winfo_t *_wpool_list_get(opal_common_ucx_wpool_t *wpool,
                                       opal_list_t *list);
static opal_common_ucx_winfo_t *_wpool_get_idle(opal_common_ucx_wpool_t *wpool,
                                       size_t comm_size);
static int _wpool_add_active(opal_common_ucx_wpool_t *wpool,
                             opal_common_ucx_winfo_t *winfo);

/* Internal Worker Pool Context management */
static void _common_ucx_wpctx_free(opal_common_ucx_ctx_t *ctx);
static int _common_ucx_wpctx_append(opal_common_ucx_ctx_t *ctx,
                                    opal_common_ucx_winfo_t *winfo);
static void _common_ucx_wpctx_remove(opal_common_ucx_ctx_t *ctx,
                                     opal_common_ucx_winfo_t *winfo);

/* Internal Worker Pool Memeory management */
static int _comm_ucx_wpmem_map(opal_common_ucx_wpool_t *wpool,
                               void **base, size_t size, ucp_mem_h *memh_ptr,
                               opal_common_ucx_mem_type_t mem_type);
static void _common_ucx_wpmem_free(opal_common_ucx_wpmem_t *mem);

#endif // COMMON_UCX_WPOOL_INT_H
