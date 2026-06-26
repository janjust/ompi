/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
 * Copyright (c) 2016      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2022      IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "pml_ucx_request.h"
#include "ompi/mca/pml/base/pml_base_bsend.h"
#include "ompi/message/message.h"
#include "ompi/runtime/ompi_spc.h"
#include "ompi/request/request.h"
#include <inttypes.h>


/* ---------------------------------------------------------------------------
 * Helper: release a non-persistent PML request back to the freelist, or mark
 * it detached if the UCX operation is still in flight.
 * --------------------------------------------------------------------------*/
static inline void mca_pml_ucx_req_release(mca_pml_ucx_req_t *pml_req)
{
    if (NULL != pml_req->ucx_req) {
        pml_req->detached = true;
    } else {
        mca_pml_ucx_request_reset(&pml_req->ompi);
        PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &pml_req->ompi.super);
    }
}

/* ---------------------------------------------------------------------------
 * Non-persistent request management
 * --------------------------------------------------------------------------*/

static int mca_pml_ucx_request_free(ompi_request_t **rptr)
{
    mca_pml_ucx_req_t *pml_req = (mca_pml_ucx_req_t *)*rptr;

    PML_UCX_VERBOSE(9, "free request *%p=%p", (void*)rptr, (void*)pml_req);

    *rptr = MPI_REQUEST_NULL;
    mca_pml_ucx_req_release(pml_req);
    return OMPI_SUCCESS;
}

int mca_pml_ucx_request_cancel(ompi_request_t *req, int flag)
{
    mca_pml_ucx_req_t *pml_req = (mca_pml_ucx_req_t *)req;
    if (NULL != pml_req->ucx_req) {
        ucp_request_cancel(ompi_pml_ucx.ucp_worker, pml_req->ucx_req);
    }
    return OMPI_SUCCESS;
}

#if MPI_VERSION >= 4
int mca_pml_ucx_request_cancel_send(ompi_request_t *req, int flag)
{
    mca_pml_cancel_send_callback(req, flag);
    return mca_pml_ucx_request_cancel(req, flag);
}
#endif

/* ---------------------------------------------------------------------------
 * Internal completion helpers (called by all public completion callbacks)
 * --------------------------------------------------------------------------*/

__opal_attribute_always_inline__ static inline void
mca_pml_ucx_send_completion_internal(opal_common_ucx_request_t *ucx_req,
                                     mca_pml_ucx_req_t *pml_req,
                                     ucs_status_t status)
{
    bool detached;

    PML_UCX_VERBOSE(8, "send request %p completed with status %s",
                    (void*)pml_req, ucs_status_string(status));

    mca_pml_ucx_set_send_status(&pml_req->ompi.req_status, status);
    detached = pml_req->detached;
    pml_req->ucx_req = NULL;
    ucp_request_release(ucx_req);

    if (OPAL_UNLIKELY(detached)) {
        mca_pml_ucx_request_reset(&pml_req->ompi);
        PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &pml_req->ompi.super);
    } else {
        PML_UCX_ASSERT(!(REQUEST_COMPLETE(&pml_req->ompi)));
        ompi_request_complete(&pml_req->ompi, true);
    }
}

__opal_attribute_always_inline__ static inline void
mca_pml_ucx_bsend_completion_internal(opal_common_ucx_request_t *ucx_req,
                                       ucs_status_t status)
{
    mca_pml_ucx_req_t *pml_req = (mca_pml_ucx_req_t *)ucx_req->ext_req;

    if (NULL == pml_req) {
        /* Completed before ext_req was set (immediate completion, caller frees
         * packed_data inline). */
        ucp_request_release(ucx_req);
        return;
    }

    PML_UCX_VERBOSE(8, "bsend request %p buffer %p completed with status %s",
                    (void*)pml_req, pml_req->ompi.req_complete_cb_data,
                    ucs_status_string(status));

    mca_pml_base_bsend_request_free(pml_req->ompi.req_complete_cb_data);
    pml_req->ompi.req_complete_cb_data = NULL;
    pml_req->ucx_req = NULL;
    ucp_request_release(ucx_req);
    mca_pml_ucx_request_reset(&pml_req->ompi);
    PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &pml_req->ompi.super);
}

__opal_attribute_always_inline__ static inline void
mca_pml_ucx_recv_completion_internal(opal_common_ucx_request_t *ucx_req,
                                     mca_pml_ucx_req_t *pml_req,
                                     ucs_status_t status,
                                     const ucp_tag_recv_info_t *info)
{
    bool detached;

    PML_UCX_VERBOSE(8, "receive request %p completed with status %s tag %"PRIx64" len %zu",
                    (void*)pml_req, ucs_status_string(status), info->sender_tag,
                    info->length);

    SPC_USER_OR_MPI(PML_UCX_TAG_GET_MPI_TAG(info->sender_tag), info->length,
                    OMPI_SPC_BYTES_RECEIVED_USER, OMPI_SPC_BYTES_RECEIVED_MPI);

    mca_pml_ucx_set_recv_status(&pml_req->ompi.req_status, status, info);
    detached = pml_req->detached;
    pml_req->ucx_req = NULL;
    ucp_request_release(ucx_req);

    if (OPAL_UNLIKELY(detached)) {
        mca_pml_ucx_request_reset(&pml_req->ompi);
        PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &pml_req->ompi.super);
    } else {
        PML_UCX_ASSERT(!(REQUEST_COMPLETE(&pml_req->ompi)));
        ompi_request_complete(&pml_req->ompi, true);
    }
}

/* ---------------------------------------------------------------------------
 * Public completion callbacks invoked by UCX
 *
 * For non-nbx ("nb") callbacks, the first arg is opal_common_ucx_request_t*.
 * ext_req holds the mca_pml_ucx_req_t* (set by the caller after the UCX op).
 * A NULL ext_req means the operation completed before the caller could link
 * the requests (only possible for sends that complete inline); in that case
 * the callback just releases the UCX request and returns.
 *
 * For nbx callbacks the ompi request is passed as user_data, which eliminates
 * the ext_req link-before-callback race.
 * --------------------------------------------------------------------------*/

void mca_pml_ucx_send_completion(void *request, ucs_status_t status)
{
    opal_common_ucx_request_t *ucx_req = (opal_common_ucx_request_t *)request;
    mca_pml_ucx_req_t *pml_req = (mca_pml_ucx_req_t *)ucx_req->ext_req;

    if (NULL == pml_req) {
        /* Inline-completed send; caller handles the result via NULL return. */
        ucp_request_release(ucx_req);
        return;
    }
    mca_pml_ucx_send_completion_internal(ucx_req, pml_req, status);
}

void mca_pml_ucx_send_completion_empty(void *request, ucs_status_t status)
{
    /* Blocking-send path: the WAIT_LOOP in the caller will call
     * ucp_request_free() after polling for completion. */
    (void)request;
    (void)status;
}

void mca_pml_ucx_bsend_completion(void *request, ucs_status_t status)
{
    mca_pml_ucx_bsend_completion_internal((opal_common_ucx_request_t *)request,
                                          status);
}

void mca_pml_ucx_recv_completion(void *request, ucs_status_t status,
                                 ucp_tag_recv_info_t *info)
{
    opal_common_ucx_request_t *ucx_req = (opal_common_ucx_request_t *)request;
    mca_pml_ucx_req_t *pml_req = (mca_pml_ucx_req_t *)ucx_req->ext_req;

    if (NULL == pml_req) {
        /* The callback fired before the caller set ext_req; the caller will
         * detect completion via ucp_request_test() and complete manually. */
        return;
    }
    mca_pml_ucx_recv_completion_internal(ucx_req, pml_req, status, info);
}

void mca_pml_ucx_send_nbx_completion(void *request, ucs_status_t status,
                                     void *user_data)
{
    mca_pml_ucx_send_completion_internal((opal_common_ucx_request_t *)request,
                                         (mca_pml_ucx_req_t *)user_data,
                                         status);
}

void mca_pml_ucx_bsend_nbx_completion(void *request, ucs_status_t status,
                                      void *user_data)
{
    (void)user_data;
    mca_pml_ucx_bsend_completion_internal((opal_common_ucx_request_t *)request,
                                          status);
}

void mca_pml_ucx_recv_nbx_completion(void *request, ucs_status_t status,
                                     const ucp_tag_recv_info_t *info,
                                     void *user_data)
{
    mca_pml_ucx_recv_completion_internal((opal_common_ucx_request_t *)request,
                                         (mca_pml_ucx_req_t *)user_data,
                                         status, info);
}

/* ---------------------------------------------------------------------------
 * Persistent request support
 * --------------------------------------------------------------------------*/

static void mca_pml_ucx_persistent_request_detach(mca_pml_ucx_persistent_request_t *preq,
                                                  ompi_request_t *tmp_req)
{
    tmp_req->req_complete_cb_data = NULL;
    preq->tmp_req                 = NULL;
}

inline void
mca_pml_ucx_persistent_request_complete(mca_pml_ucx_persistent_request_t *preq,
                                        ompi_request_t *tmp_req)
{
    preq->ompi.req_status = tmp_req->req_status;
    mca_pml_ucx_request_reset(tmp_req);
    mca_pml_ucx_persistent_request_detach(preq, tmp_req);
    PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &tmp_req->super);
    ompi_request_complete(&preq->ompi, true);
}

static inline void mca_pml_ucx_preq_completion(ompi_request_t *tmp_req)
{
    mca_pml_ucx_persistent_request_t *preq;
    mca_pml_ucx_req_t *pml_req = (mca_pml_ucx_req_t *)tmp_req;

    ompi_request_complete(tmp_req, false);
    preq = (mca_pml_ucx_persistent_request_t*)tmp_req->req_complete_cb_data;
    if (preq != NULL) {
        PML_UCX_ASSERT(preq->tmp_req != NULL);
        mca_pml_ucx_persistent_request_complete(preq, tmp_req);
    } else if (pml_req->detached) {
        /* MPI_Request_free was called on the persistent request while the
         * tmp_req operation was still in flight. */
        mca_pml_ucx_request_reset(tmp_req);
        PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &tmp_req->super);
    }
}

void mca_pml_ucx_psend_completion(void *request, ucs_status_t status)
{
    opal_common_ucx_request_t *ucx_req = (opal_common_ucx_request_t *)request;
    mca_pml_ucx_req_t *pml_req = (mca_pml_ucx_req_t *)ucx_req->ext_req;

    if (NULL == pml_req) {
        /* Immediate send completion before ext_req was linked. */
        ucp_request_release(ucx_req);
        return;
    }

    PML_UCX_VERBOSE(8, "persistent send request %p completed with status %s",
                    (void*)pml_req, ucs_status_string(status));

    mca_pml_ucx_set_send_status(&pml_req->ompi.req_status, status);
    pml_req->ucx_req = NULL;
    ucp_request_release(ucx_req);
    mca_pml_ucx_preq_completion(&pml_req->ompi);
}

void mca_pml_ucx_precv_completion(void *request, ucs_status_t status,
                                  ucp_tag_recv_info_t *info)
{
    opal_common_ucx_request_t *ucx_req = (opal_common_ucx_request_t *)request;
    mca_pml_ucx_req_t *pml_req = (mca_pml_ucx_req_t *)ucx_req->ext_req;

    if (NULL == pml_req) {
        /* Callback fired before ext_req was set; caller detects via
         * ucp_request_test() and handles manually. */
        return;
    }

    PML_UCX_VERBOSE(8, "persistent receive request %p completed with status %s "
                    "tag %"PRIx64" len %zu",
                    (void*)pml_req, ucs_status_string(status), info->sender_tag,
                    info->length);

    mca_pml_ucx_set_recv_status(&pml_req->ompi.req_status, status, info);
    pml_req->ucx_req = NULL;
    ucp_request_release(ucx_req);
    mca_pml_ucx_preq_completion(&pml_req->ompi);
}

/* ---------------------------------------------------------------------------
 * Request class constructors / destructors
 * --------------------------------------------------------------------------*/

static void mca_pml_ucx_request_init_common(ompi_request_t* ompi_req,
                                            bool req_persistent,
                                            ompi_request_state_t state,
                                            ompi_request_free_fn_t req_free,
                                            ompi_request_cancel_fn_t req_cancel)
{
    OMPI_REQUEST_INIT(ompi_req, req_persistent);
    ompi_req->req_type             = OMPI_REQUEST_PML;
    ompi_req->req_state            = state;
    ompi_req->req_start            = mca_pml_ucx_start;
    ompi_req->req_free             = req_free;
    ompi_req->req_cancel           = req_cancel;
    ompi_req->req_complete_cb_data = NULL;
}

static void mca_pml_ucx_req_construct(mca_pml_ucx_req_t *req)
{
    mca_pml_ucx_request_init_common(&req->ompi, false, OMPI_REQUEST_ACTIVE,
                                    mca_pml_ucx_request_free,
                                    mca_pml_ucx_request_cancel);
    req->ucx_req  = NULL;
    req->detached = false;
}

static void mca_pml_ucx_req_destruct(mca_pml_ucx_req_t *req)
{
    req->ompi.req_state = OMPI_REQUEST_INVALID;
    OMPI_REQUEST_FINI(&req->ompi);
}

OBJ_CLASS_INSTANCE(mca_pml_ucx_req_t,
                   ompi_request_t,
                   mca_pml_ucx_req_construct,
                   mca_pml_ucx_req_destruct);

static int mca_pml_ucx_persistent_request_free(ompi_request_t **rptr)
{
    mca_pml_ucx_persistent_request_t* preq = (mca_pml_ucx_persistent_request_t*)*rptr;
    ompi_request_t *tmp_req = preq->tmp_req;

    preq->ompi.req_state = OMPI_REQUEST_INVALID;
    if (tmp_req != NULL) {
        mca_pml_ucx_persistent_request_detach(preq, tmp_req);
        mca_pml_ucx_req_release((mca_pml_ucx_req_t*)tmp_req);
    }
    OMPI_DATATYPE_RELEASE(preq->ompi_datatype);
    PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.persistent_reqs, &preq->ompi.super);
    *rptr = MPI_REQUEST_NULL;
    return OMPI_SUCCESS;
}

static int mca_pml_ucx_persistent_request_cancel(ompi_request_t *req, int flag)
{
    mca_pml_ucx_persistent_request_t* preq = (mca_pml_ucx_persistent_request_t*)req;

    if (preq->tmp_req != NULL) {
        mca_pml_ucx_req_t *pml_req = (mca_pml_ucx_req_t*)preq->tmp_req;
        if (NULL != pml_req->ucx_req) {
            ucp_request_cancel(ompi_pml_ucx.ucp_worker, pml_req->ucx_req);
        }
    }
    return OMPI_SUCCESS;
}

static void mca_pml_ucx_persisternt_request_construct(mca_pml_ucx_persistent_request_t* req)
{
    mca_pml_ucx_request_init_common(&req->ompi, true, OMPI_REQUEST_INACTIVE,
                                    mca_pml_ucx_persistent_request_free,
                                    mca_pml_ucx_persistent_request_cancel);
    req->tmp_req = NULL;
}

static void mca_pml_ucx_persisternt_request_destruct(mca_pml_ucx_persistent_request_t* req)
{
    req->ompi.req_state = OMPI_REQUEST_INVALID;
    OMPI_REQUEST_FINI(&req->ompi);
}

OBJ_CLASS_INSTANCE(mca_pml_ucx_persistent_request_t,
                   ompi_request_t,
                   mca_pml_ucx_persisternt_request_construct,
                   mca_pml_ucx_persisternt_request_destruct);

static int mca_pml_completed_request_free(struct ompi_request_t** rptr)
{
    *rptr = MPI_REQUEST_NULL;
    return OMPI_SUCCESS;
}

static int mca_pml_completed_request_cancel(struct ompi_request_t* ompi_req, int flag)
{
    return OMPI_SUCCESS;
}

void mca_pml_ucx_completed_request_init(ompi_request_t *ompi_req)
{
    mca_pml_ucx_request_init_common(ompi_req, false, OMPI_REQUEST_ACTIVE,
                                    mca_pml_completed_request_free,
                                    mca_pml_completed_request_cancel);
    ompi_req->req_mpi_object.comm = &ompi_mpi_comm_world.comm;
    ompi_request_complete(ompi_req, false);
}
