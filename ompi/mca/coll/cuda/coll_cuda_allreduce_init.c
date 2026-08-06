/*
 * Copyright (c) 2014-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2014-2026 NVIDIA Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * Persistent allreduce wrapper for the coll/cuda module.
 *
 * coll/cuda provides mca_coll_cuda_allreduce for blocking allreduce, which
 * detects CUDA device buffers and performs D2H / H2D staging around the
 * downchain blocking collective.  However, the module previously registered
 * no coll_allreduce_init handler, so persistent allreduce fell through to
 * coll/libnbc.  libnbc builds an NBC schedule that stores the raw device
 * pointers and later runs CPU reduce/copy operations on them inside
 * NBC_Progress, causing a SIGSEGV (SI_ACCERR) at MPI_Wait time.
 *
 * Fix: register mca_coll_cuda_allreduce_init.  It creates a custom
 * persistent request whose req_start callback calls mca_coll_cuda_allreduce
 * synchronously, reusing the existing staging logic, then marks the request
 * complete.  MPI_Wait returns immediately because completion is already set.
 */

#include "ompi_config.h"
#include "coll_cuda.h"

#include "ompi/request/request.h"
#include "ompi/mca/coll/base/coll_base_util.h"

/* ------------------------------------------------------------------ */
/* Internal persistent-request type                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    ompi_coll_base_nbc_request_t super;
    const void               *sendbuf;
    void                     *recvbuf;
    int                       count;
    ompi_datatype_t          *dtype;
    ompi_op_t                *op;
    ompi_communicator_t      *comm;
    mca_coll_cuda_module_t   *cuda_module;
} mca_coll_cuda_allreduce_request_t;

/* Forward declarations for the request callbacks. */
static int cuda_allreduce_req_start(size_t count, ompi_request_t **reqs);
static int cuda_allreduce_req_free(ompi_request_t **req);
static int cuda_allreduce_req_cancel(ompi_request_t *req, int complete);

static void
cuda_allreduce_request_construct(mca_coll_cuda_allreduce_request_t *req)
{
    req->super.super.req_type       = OMPI_REQUEST_COLL;
    req->super.super.req_persistent = true;
    req->super.super.req_start      = cuda_allreduce_req_start;
    req->super.super.req_free       = cuda_allreduce_req_free;
    req->super.super.req_cancel     = cuda_allreduce_req_cancel;
    req->super.super.req_status._cancelled = 0;
}

OBJ_CLASS_INSTANCE(mca_coll_cuda_allreduce_request_t,
                   ompi_coll_base_nbc_request_t,
                   cuda_allreduce_request_construct,
                   NULL);

/* ------------------------------------------------------------------ */
/* Request callbacks                                                   */
/* ------------------------------------------------------------------ */

/*
 * Called by MPI_Start / MPI_Startall.  Drives the full allreduce
 * synchronously via mca_coll_cuda_allreduce, which performs D2H/H2D
 * staging when either buffer resides on an accelerator device.
 */
static int
cuda_allreduce_req_start(size_t count, ompi_request_t **reqs)
{
    for (size_t i = 0; i < count; i++) {
        mca_coll_cuda_allreduce_request_t *req =
            (mca_coll_cuda_allreduce_request_t *) reqs[i];
        int rc;

        req->super.super.req_complete = REQUEST_PENDING;

        rc = mca_coll_cuda_allreduce(req->sendbuf, req->recvbuf,
                                     req->count, req->dtype, req->op,
                                     req->comm, &req->cuda_module->super);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != rc)) {
            return rc;
        }

        ompi_request_complete(&req->super.super, true);
    }

    return OMPI_SUCCESS;
}

static int
cuda_allreduce_req_free(ompi_request_t **req)
{
    OBJ_RELEASE(*req);
    *req = MPI_REQUEST_NULL;
    return OMPI_SUCCESS;
}

static int
cuda_allreduce_req_cancel(ompi_request_t *req, int complete)
{
    (void) req;
    (void) complete;
    return MPI_ERR_REQUEST;
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                  */
/* ------------------------------------------------------------------ */

int
mca_coll_cuda_allreduce_init(const void *sbuf, void *rbuf, int count,
                              struct ompi_datatype_t *dtype,
                              struct ompi_op_t *op,
                              struct ompi_communicator_t *comm,
                              struct ompi_info_t *info,
                              ompi_request_t **request,
                              mca_coll_base_module_t *module)
{
    mca_coll_cuda_allreduce_request_t *req;

    (void) info; /* info is not used for this collective */

    req = OBJ_NEW(mca_coll_cuda_allreduce_request_t);
    if (OPAL_UNLIKELY(NULL == req)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    OMPI_REQUEST_INIT(&req->super.super, true /* persistent */);

    req->sendbuf     = sbuf;
    req->recvbuf     = rbuf;
    req->count       = count;
    req->dtype       = dtype;
    req->op          = op;
    req->comm        = comm;
    req->cuda_module = (mca_coll_cuda_module_t *) module;

    *request = &req->super.super;
    return OMPI_SUCCESS;
}
