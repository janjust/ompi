/*
 * Copyright (C) 2001-2011 Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
 * Copyright (c) 2016-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2018-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2018-2022 IBM Corporation.  All rights reserved.
 * Copyright (c) 2019      Intel, Inc.  All rights reserved.
 * Copyright (c) 2022      Amazon.com, Inc. or its affiliates.
 *                         All Rights reserved.
 * Copyright (c) 2026      Triad National Security, LLC. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "pml_ucx.h"

#include "opal/runtime/opal.h"
#include "opal/mca/pmix/pmix-internal.h"
#include "ompi/attribute/attribute.h"
#include "ompi/message/message.h"
#include "ompi/runtime/ompi_spc.h"
#include "ompi/mca/pml/base/pml_base_bsend.h"
#include "opal/mca/common/ucx/common_ucx.h"
#include "pml_ucx_request.h"

#include <inttypes.h>

/*
 * Maximum number of transport entries that we can expect from UCX
 */
#define PML_UCX_MAX_TRANSPORT_ENTRIES 100

#define PML_UCX_TRACE_SEND(_msg, _buf, _count, _datatype, _dst, _tag, _mode, _comm, ...) \
    PML_UCX_VERBOSE(8, _msg " buf %p count %zu type '%s' dst %d tag %d mode %s comm %d '%s'", \
                    __VA_ARGS__, \
                    (_buf), (_count), (_datatype)->name, (_dst), (_tag), \
                    mca_pml_ucx_send_mode_name(_mode), (_comm)->c_index, \
                    (_comm)->c_name);

#define PML_UCX_TRACE_RECV(_msg, _buf, _count, _datatype, _src, _tag, _comm, ...) \
    PML_UCX_VERBOSE(8, _msg " buf %p count %zu type '%s' src %d tag %d comm %d '%s'", \
                    __VA_ARGS__, \
                    (_buf), (_count), (_datatype)->name, (_src), (_tag), \
                    (_comm)->c_index, (_comm)->c_name);

#define PML_UCX_TRACE_PROBE(_msg, _src, _tag, _comm) \
    PML_UCX_VERBOSE(8, _msg " src %d tag %d comm %d '%s'", \
                    _src, (_tag), (_comm)->c_index, (_comm)->c_name);

#define PML_UCX_TRACE_MRECV(_msg, _buf, _count, _datatype, _message) \
    PML_UCX_VERBOSE(8, _msg " buf %p count %zu type '%s' msg *%p=%p (%p)", \
                    (_buf), (_count), (_datatype)->name, (void*)(_message), \
                    (void*)*(_message), (*(_message))->req_ptr);

mca_pml_transports_t *mca_pml_ucx_get_transports(ompi_communicator_t *comm,
                                                 int rank);

#define MODEX_KEY "pml-ucx"

mca_pml_ucx_module_t ompi_pml_ucx = {
    .super = {
        .pml_add_procs      = mca_pml_ucx_add_procs,
        .pml_del_procs      = mca_pml_ucx_del_procs,
        .pml_enable         = mca_pml_ucx_enable,
        .pml_progress       = NULL,
        .pml_add_comm       = mca_pml_ucx_add_comm,
        .pml_del_comm       = mca_pml_ucx_del_comm,
        .pml_irecv_init     = mca_pml_ucx_irecv_init,
        .pml_irecv          = mca_pml_ucx_irecv,
        .pml_recv           = mca_pml_ucx_recv,
        .pml_isend_init     = mca_pml_ucx_isend_init,
        .pml_isend          = mca_pml_ucx_isend,
        .pml_send           = mca_pml_ucx_send,
        .pml_iprobe         = mca_pml_ucx_iprobe,
        .pml_probe          = mca_pml_ucx_probe,
        .pml_start          = mca_pml_ucx_start,
        .pml_improbe        = mca_pml_ucx_improbe,
        .pml_mprobe         = mca_pml_ucx_mprobe,
        .pml_imrecv         = mca_pml_ucx_imrecv,
        .pml_mrecv          = mca_pml_ucx_mrecv,
        .pml_dump           = mca_pml_ucx_dump,
        .pml_max_contextid  = (1ul << (PML_UCX_CONTEXT_BITS)) - 1,
        .pml_max_tag        = (1ul << (PML_UCX_TAG_BITS - 1)) - 1,
        .pml_flags          = 0, /* flags */
        .pml_get_transports = mca_pml_ucx_get_transports
    },
    .wpool                 = NULL,
    .ucp_context           = NULL,
    .ucp_worker            = NULL
};

#define PML_UCX_REQ_ALLOCA() \
    ((char *)alloca(ompi_pml_ucx.request_size) + ompi_pml_ucx.request_size);

#if HAVE_UCP_WORKER_ADDRESS_FLAGS
static int mca_pml_ucx_send_worker_address_type(int addr_flags, int modex_scope)
{
    ucs_status_t status;
    ucp_worker_attr_t attrs;
    int rc;

    attrs.field_mask    = UCP_WORKER_ATTR_FIELD_ADDRESS |
                          UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
    attrs.address_flags = addr_flags;

    status = ucp_worker_query(ompi_pml_ucx.ucp_worker, &attrs);
    if (UCS_OK != status) {
        PML_UCX_ERROR("Failed to query UCP worker address");
        return OMPI_ERROR;
    }

    OPAL_MODEX_SEND(rc, modex_scope, &mca_pml_ucx_component.pmlm_version,
                    (void*)attrs.address, attrs.address_length);

    ucp_worker_release_address(ompi_pml_ucx.ucp_worker, attrs.address);

    if (OMPI_SUCCESS != rc) {
        return OMPI_ERROR;
    }

    PML_UCX_VERBOSE(2, "Pack %s worker address, size %ld",
                    (modex_scope == PMIX_LOCAL) ? "local" : "remote",
                    attrs.address_length);

    return OMPI_SUCCESS;
}
#endif

static int mca_pml_ucx_send_worker_address(void)
{
    ucs_status_t status;

#if !HAVE_UCP_WORKER_ADDRESS_FLAGS
    ucp_address_t *address;
    size_t addrlen;
    int rc;

    status = ucp_worker_get_address(ompi_pml_ucx.ucp_worker, &address, &addrlen);
    if (UCS_OK != status) {
        PML_UCX_ERROR("Failed to get worker address");
        return OMPI_ERROR;
    }

    PML_UCX_VERBOSE(2, "Pack worker address, size %ld", addrlen);

    OPAL_MODEX_SEND(rc, PMIX_GLOBAL,
                    &mca_pml_ucx_component.pmlm_version, (void*)address, addrlen);

    ucp_worker_release_address(ompi_pml_ucx.ucp_worker, address);

    if (OMPI_SUCCESS != rc) {
        goto err;
    }
#else
    /* Pack just network device addresses for remote node peers */
    status = mca_pml_ucx_send_worker_address_type(UCP_WORKER_ADDRESS_FLAG_NET_ONLY,
                                                  PMIX_REMOTE);
    if (UCS_OK != status) {
        goto err;
    }

    status = mca_pml_ucx_send_worker_address_type(0, PMIX_LOCAL);
    if (UCS_OK != status) {
        goto err;
    }
#endif

    return OMPI_SUCCESS;

err:
    PML_UCX_ERROR("Open MPI couldn't distribute EP connection details");
    return OMPI_ERROR;
}

static int mca_pml_ucx_recv_worker_address(ompi_proc_t *proc,
                                           ucp_address_t **address_p,
                                           size_t *addrlen_p)
{
    int ret;

    *address_p = NULL;
    OPAL_MODEX_RECV(ret, &mca_pml_ucx_component.pmlm_version, &proc->super.proc_name,
                    (void**)address_p, addrlen_p);
    if (ret < 0) {
        PML_UCX_ERROR("Failed to receive UCX worker address: %s (%d)",
                      opal_strerror(ret), ret);
    }

    PML_UCX_VERBOSE(2, "Got proc %d address, size %ld",
                    proc->super.proc_name.vpid, *addrlen_p);
    return ret;
}

int mca_pml_ucx_open(void)
{
    unsigned major_version, minor_version, release_number;

    /* Check version */
    ucp_get_version(&major_version, &minor_version, &release_number);
    PML_UCX_VERBOSE(1, "mca_pml_ucx_open: UCX version %u.%u.%u",
                    major_version, minor_version, release_number);

    if ((major_version == 1) && (minor_version == 8)) {
        /* disabled due to issue #8321 */
        PML_UCX_VERBOSE(1, "UCX PML is disabled because the run-time UCX version "
                           "is 1.8, which has a known catastrophic issue");
        return OMPI_ERROR;
    }

    if ((major_version == 1) && (minor_version < 9)) {
        /* show warning due to issue #8549 */
        PML_UCX_WARN("UCX version %u.%u.%u is too old, please install 1.9.x or "
                     "newer", major_version, minor_version, release_number);
    }

    /* Register the UCP features PML requires in the shared UCX context.
     * The context is created later in mca_pml_ucx_init() once all components
     * have registered their features via opal_common_ucx_add_features(). */
    opal_common_ucx_add_features(UCP_FEATURE_TAG);

    return OMPI_SUCCESS;
}

int mca_pml_ucx_close(void)
{
    PML_UCX_VERBOSE(1, "mca_pml_ucx_close");
    /* UCP context is owned by the global shared wpool; do not clean up here. */
    ompi_pml_ucx.ucp_context = NULL;
    return OMPI_SUCCESS;
}

int mca_pml_ucx_init(int enable_mpi_threads)
{
    opal_common_ucx_wpool_t *wpool;
    ucp_worker_attr_t attr;
    ucp_context_attr_t ctx_attr;
    ucs_status_t status;
    int i, rc;

    PML_UCX_VERBOSE(1, "mca_pml_ucx_init");

    /* Obtain (or create) the global shared UCX context + worker pool.
     * All components have called opal_common_ucx_add_features() in their
     * open() callbacks, so the accumulated feature set is complete here. */
    wpool = opal_common_ucx_global_wpool_get(enable_mpi_threads,
                                             ompi_proc_world_size());
    if (NULL == wpool) {
        PML_UCX_ERROR("Failed to get global UCX wpool");
        return OMPI_ERROR;
    }
    ompi_pml_ucx.wpool       = wpool;
    ompi_pml_ucx.ucp_context = wpool->ucp_ctx;
    ompi_pml_ucx.ucp_worker  = wpool->dflt_winfo->worker;

    /* Query the shared context for the actual request size (UCX internal
     * overhead + user portion) used by PML_UCX_REQ_ALLOCA(). */
    ctx_attr.field_mask = UCP_ATTR_FIELD_REQUEST_SIZE;
    status = ucp_context_query(wpool->ucp_ctx, &ctx_attr);
    if (UCS_OK != status) {
        PML_UCX_ERROR("Failed to query UCP context attributes");
        rc = OMPI_ERROR;
        goto err_wpool_put;
    }
    ompi_pml_ucx.request_size = ctx_attr.request_size;

    /* Verify that the shared worker thread mode satisfies PML requirements. */
    attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
    status = ucp_worker_query(ompi_pml_ucx.ucp_worker, &attr);
    if (UCS_OK != status) {
        PML_UCX_ERROR("Failed to query UCP worker thread level");
        rc = OMPI_ERROR;
        goto err_wpool_put;
    }

    if (enable_mpi_threads && (attr.thread_mode != UCS_THREAD_MODE_MULTI)) {
        /* UCX does not support multithreading, disqualify current PML for now */
        /* TODO: we should let OMPI to fallback to THREAD_SINGLE mode */
        PML_UCX_WARN("UCP worker does not support MPI_THREAD_MULTIPLE. "
                     "PML UCX could not be selected");
        rc = OMPI_ERR_NOT_SUPPORTED;
        goto err_wpool_put;
    }

    rc = mca_pml_ucx_send_worker_address();
    if (rc < 0) {
        goto err_wpool_put;
    }

    ompi_pml_ucx.datatype_attr_keyval = MPI_KEYVAL_INVALID;
    for (i = 0; i < OMPI_DATATYPE_MAX_PREDEFINED; ++i) {
        ompi_pml_ucx.predefined_types[i] = PML_UCX_DATATYPE_INVALID;
    }

    /* Initialize the free lists */
    OBJ_CONSTRUCT(&ompi_pml_ucx.persistent_reqs, mca_pml_ucx_freelist_t);
    OBJ_CONSTRUCT(&ompi_pml_ucx.reqs,            mca_pml_ucx_freelist_t);
    OBJ_CONSTRUCT(&ompi_pml_ucx.convs,           mca_pml_ucx_freelist_t);

    /* Create a completed request to be returned from isend */
    OBJ_CONSTRUCT(&ompi_pml_ucx.completed_send_req, ompi_request_t);
    mca_pml_ucx_completed_request_init(&ompi_pml_ucx.completed_send_req);
#if MPI_VERSION >= 4
    ompi_pml_ucx.completed_send_req.req_cancel = mca_pml_cancel_send_callback;
#endif

    opal_progress_register(mca_pml_ucx_progress);

    PML_UCX_VERBOSE(2, "using shared ucp context %p, worker %p",
                    (void *)ompi_pml_ucx.ucp_context,
                    (void *)ompi_pml_ucx.ucp_worker);
    return OMPI_SUCCESS;

err_wpool_put:
    opal_common_ucx_global_wpool_put();
    ompi_pml_ucx.wpool      = NULL;
    ompi_pml_ucx.ucp_context = NULL;
    ompi_pml_ucx.ucp_worker  = NULL;
    return rc;
}

int mca_pml_ucx_cleanup(void)
{
    int i;

    PML_UCX_VERBOSE(1, "mca_pml_ucx_cleanup");

    opal_progress_unregister(mca_pml_ucx_progress);

    if (ompi_pml_ucx.datatype_attr_keyval != MPI_KEYVAL_INVALID) {
        ompi_attr_free_keyval(TYPE_ATTR, &ompi_pml_ucx.datatype_attr_keyval, false);
    }

    for (i = 0; i < OMPI_DATATYPE_MAX_PREDEFINED; ++i) {
        if (ompi_pml_ucx.predefined_types[i] != PML_UCX_DATATYPE_INVALID) {
            ucp_dt_destroy(ompi_pml_ucx.predefined_types[i]);
            ompi_pml_ucx.predefined_types[i] = PML_UCX_DATATYPE_INVALID;
        }
    }

    ompi_pml_ucx.completed_send_req.req_state = OMPI_REQUEST_INVALID;
    OMPI_REQUEST_FINI(&ompi_pml_ucx.completed_send_req);
    OBJ_DESTRUCT(&ompi_pml_ucx.completed_send_req);

    OBJ_DESTRUCT(&ompi_pml_ucx.convs);
    OBJ_DESTRUCT(&ompi_pml_ucx.reqs);
    OBJ_DESTRUCT(&ompi_pml_ucx.persistent_reqs);

    /* Worker and context are owned by the global shared wpool; release our
     * reference instead of destroying them directly. */
    ompi_pml_ucx.ucp_worker  = NULL;
    ompi_pml_ucx.ucp_context = NULL;
    if (NULL != ompi_pml_ucx.wpool) {
        opal_common_ucx_global_wpool_put();
        ompi_pml_ucx.wpool = NULL;
    }

    return OMPI_SUCCESS;
}

static ucp_ep_h mca_pml_ucx_add_proc_common(ompi_proc_t *proc)
{
    size_t addrlen = 0;
    ucp_ep_params_t ep_params;
    ucp_address_t *address;
    ucs_status_t status;
    ucp_ep_h ep;
    int ret;

    /* Do not add a new endpoint if we already created one */
    if (NULL != proc->proc_endpoints[OMPI_PROC_ENDPOINT_TAG_PML]) {
        return proc->proc_endpoints[OMPI_PROC_ENDPOINT_TAG_PML];
    }

    ret = mca_pml_ucx_recv_worker_address(proc, &address, &addrlen);
    if (ret < 0) {
        return NULL;
    }

    PML_UCX_VERBOSE(2, "connecting to proc. %d", proc->super.proc_name.vpid);

    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = address;

    status = ucp_ep_create(ompi_pml_ucx.ucp_worker, &ep_params, &ep);
    free(address);
    if (UCS_OK != status) {
        PML_UCX_ERROR("ucp_ep_create(proc=%d) failed: %s",
                      proc->super.proc_name.vpid,
                      ucs_status_string(status));
        return NULL;
    }

    proc->proc_endpoints[OMPI_PROC_ENDPOINT_TAG_PML] = ep;
    return ep;
}

int mca_pml_ucx_add_procs(struct ompi_proc_t **procs, size_t nprocs)
{
    ompi_proc_t *proc;
    ucp_ep_h ep;
    size_t i;

    for (i = 0; i < nprocs; ++i) {
        proc = procs[(i + OMPI_PROC_MY_NAME->vpid) % nprocs];
        ep = mca_pml_ucx_add_proc_common(proc);
        if (ep == NULL) {
            return OMPI_ERROR;
        }
    }

    opal_common_ucx_mca_proc_added();
    return OMPI_SUCCESS;
}

mca_pml_transports_t *mca_pml_ucx_get_transports(struct ompi_communicator_t *comm,
                                                 int rank)
{
#if HAVE_DECL_UCP_EP_ATTR_FIELD_TRANSPORTS
    mca_pml_transports_t *transports;
    unsigned i;
    ucs_status_t status;
    ucp_ep_attr_t ep_attrs;
    ompi_proc_t *proc = ompi_comm_peer_lookup(comm, rank);

    if (NULL == proc) {
        return NULL;
    }
    if (NULL == proc->proc_endpoints[OMPI_PROC_ENDPOINT_TAG_PML]) {
        return NULL;
    }

    ep_attrs.field_mask = UCP_EP_ATTR_FIELD_TRANSPORTS;
    ep_attrs.transports.num_entries = PML_UCX_MAX_TRANSPORT_ENTRIES;
    ep_attrs.transports.entry_size = sizeof(ucp_transport_entry_t);
    ep_attrs.transports.entries = (ucp_transport_entry_t *) malloc(
             sizeof(ucp_transport_entry_t) * ep_attrs.transports.num_entries);
    if (NULL == ep_attrs.transports.entries) {
        return NULL;
    }

    status = ucp_ep_query(proc->proc_endpoints[OMPI_PROC_ENDPOINT_TAG_PML],
                          &ep_attrs);
    if (UCS_OK != status) {
        return NULL;
    }

    transports = (mca_pml_transports_t *) malloc(sizeof(mca_pml_transports_t));
    if (NULL == transports) {
        return NULL;
    }

    transports->entries = (mca_pml_transport_entry_t *) malloc(
        ep_attrs.transports.num_entries * sizeof(mca_pml_transport_entry_t));
    if (NULL == transports->entries) {
        free(transports);
        return NULL;
    }

    for (i = 0; i < ep_attrs.transports.num_entries; i++) {
        transports->entries[i].transport_name =
                ep_attrs.transports.entries[i].transport_name;
        transports->entries[i].device_name = 
                ep_attrs.transports.entries[i].device_name;
    }
    transports->count = ep_attrs.transports.num_entries;

    free(ep_attrs.transports.entries);
    return transports;
#else
    return NULL;
#endif
}

__opal_attribute_always_inline__
static inline ucp_ep_h mca_pml_ucx_get_ep(ompi_communicator_t *comm, int rank)
{
    ompi_proc_t *proc_peer = ompi_comm_peer_lookup(comm, rank);
    ucp_ep_h ep;

    ep = proc_peer->proc_endpoints[OMPI_PROC_ENDPOINT_TAG_PML];
    if (OPAL_LIKELY(NULL != ep)) {
        return ep;
    }

    return mca_pml_ucx_add_proc_common(proc_peer);
}

int mca_pml_ucx_del_procs(struct ompi_proc_t **procs, size_t nprocs)
{
    ompi_proc_t *proc;
    opal_common_ucx_del_proc_t *del_procs;
    size_t i;
    int ret;

    del_procs = malloc(sizeof(*del_procs) * nprocs);
    if (del_procs == NULL) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    for (i = 0; i < nprocs; ++i) {
        proc = procs[i];
        del_procs[i].ep   = proc->proc_endpoints[OMPI_PROC_ENDPOINT_TAG_PML];
        del_procs[i].vpid = proc->super.proc_name.vpid;

        /* mark peer as disconnected */
        proc->proc_endpoints[OMPI_PROC_ENDPOINT_TAG_PML] = NULL;
    }

    ret = opal_common_ucx_del_procs(del_procs, nprocs, OMPI_PROC_MY_NAME->vpid,
                                    ompi_pml_ucx.num_disconnect, ompi_pml_ucx.ucp_worker);
    free(del_procs);

    return ret;
}

int mca_pml_ucx_enable(bool enable)
{
    ompi_attribute_fn_ptr_union_t copy_fn;
    ompi_attribute_fn_ptr_union_t del_fn;
    int ret;

    /* Create a key for adding custom attributes to datatypes */
    copy_fn.attr_datatype_copy_fn  = MPI_TYPE_NULL_COPY_FN;
    del_fn.attr_datatype_delete_fn = mca_pml_ucx_datatype_attr_del_fn;
    ret = ompi_attr_create_keyval(TYPE_ATTR, copy_fn, del_fn,
                                  &ompi_pml_ucx.datatype_attr_keyval, NULL, 0,
                                  NULL);
    if (ret != OMPI_SUCCESS) {
        PML_UCX_ERROR("Failed to create keyval for UCX datatypes: %d", ret);
        return ret;
    }

    PML_UCX_FREELIST_INIT(&ompi_pml_ucx.persistent_reqs,
                          mca_pml_ucx_persistent_request_t,
                          128, -1, 128);
    PML_UCX_FREELIST_INIT(&ompi_pml_ucx.reqs,
                          mca_pml_ucx_req_t,
                          128, -1, 128);
    PML_UCX_FREELIST_INIT(&ompi_pml_ucx.convs,
                          mca_pml_ucx_convertor_t,
                          128, -1, 128);
    return OMPI_SUCCESS;
}

int mca_pml_ucx_progress(void)
{
    return ucp_worker_progress(ompi_pml_ucx.ucp_worker);
}

int mca_pml_ucx_add_comm(struct ompi_communicator_t* comm)
{
    return OMPI_SUCCESS;
}

int mca_pml_ucx_del_comm(struct ompi_communicator_t* comm)
{
    return OMPI_SUCCESS;
}

int mca_pml_ucx_irecv_init(void *buf, size_t count, ompi_datatype_t *datatype,
                             int src, int tag, struct ompi_communicator_t* comm,
                             struct ompi_request_t **request)
{
    mca_pml_ucx_persistent_request_t *req;

    req = (mca_pml_ucx_persistent_request_t *)PML_UCX_FREELIST_GET(&ompi_pml_ucx.persistent_reqs);
    if (req == NULL) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    PML_UCX_TRACE_RECV("irecv_init request *%p=%p", buf, count, datatype, src,
                       tag, comm, (void*)request, (void*)req);

    req->ompi.req_state           = OMPI_REQUEST_INACTIVE;
    req->ompi.req_mpi_object.comm = comm;
    req->flags                    = 0;
    req->buffer                   = buf;
    req->count                    = count;
    req->ompi_datatype            = datatype;
    req->datatype                 = mca_pml_ucx_get_datatype(datatype);
    OMPI_DATATYPE_RETAIN(datatype);

    PML_UCX_MAKE_RECV_TAG(req->tag, req->recv.tag_mask, tag, src, comm);

    *request = &req->ompi;
    return OMPI_SUCCESS;
}

/* Helper shared by irecv, imrecv, mrecv, and start (persistent recv).
 * Posts a UCX receive, links the returned UCX request to a freelist
 * pml_req, and detects immediate completion via ucp_request_test so that
 * callers see a consistent req_complete state regardless of when the UCX
 * completion callback fires relative to the ext_req assignment. */
static inline int
mca_pml_ucx_post_recv(opal_common_ucx_request_t *ucx_req,
                      mca_pml_ucx_req_t *pml_req,
                      struct ompi_communicator_t *comm)
{
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(ucx_req))) {
        PML_UCX_ERROR("ucx recv failed: %s",
                      ucs_status_string(UCS_PTR_STATUS(ucx_req)));
        PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &pml_req->ompi.super);
        return OMPI_ERROR;
    }

    PML_UCX_VERBOSE(8, "got pml request %p ucx_req %p",
                    (void*)pml_req, (void*)ucx_req);

    pml_req->ompi.req_mpi_object.comm = comm;
    ucx_req->ext_req = pml_req;
    pml_req->ucx_req = ucx_req;

    /* If the message was already in the unexpected queue, UCX may have called
     * the completion callback before we linked ext_req (callback checks for
     * NULL and returns early).  Use ucp_request_test to detect this case and
     * complete the pml_req manually with the recv info. */
    status = ucp_request_test(ucx_req, &info);
    if (UCS_INPROGRESS != status) {
        if (!REQUEST_COMPLETE(&pml_req->ompi)) {
            /* Callback fired before ext_req was linked — complete now. */
            mca_pml_ucx_set_recv_status(&pml_req->ompi.req_status, status, &info);
            pml_req->ucx_req = NULL;
            ucp_request_release(ucx_req);
            ompi_request_complete(&pml_req->ompi, true);
        }
        /* else: callback fired after ext_req was linked — already complete. */
    }
    return OMPI_SUCCESS;
}

int mca_pml_ucx_irecv(void *buf, size_t count, ompi_datatype_t *datatype,
                      int src, int tag, struct ompi_communicator_t* comm,
                      struct ompi_request_t **request)
{
    mca_pml_ucx_req_t *pml_req;
    opal_common_ucx_request_t *ucx_req;
    ucp_tag_t ucp_tag, ucp_tag_mask;
    int rc;

    PML_UCX_TRACE_RECV("irecv request *%p", buf, count, datatype, src, tag, comm,
                       (void*)request);

    pml_req = (mca_pml_ucx_req_t *)PML_UCX_FREELIST_GET(&ompi_pml_ucx.reqs);
    if (OPAL_UNLIKELY(NULL == pml_req)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    mca_pml_ucx_request_reset(&pml_req->ompi);
    pml_req->ucx_req  = NULL;
    pml_req->detached = false;

    PML_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src, comm);
#if HAVE_DECL_UCP_TAG_RECV_NBX
    {
        pml_ucx_datatype_t *op_data = mca_pml_ucx_get_op_data(datatype);
        ucp_request_param_t param = op_data->op_param.irecv;
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_USER_DATA;
        param.user_data     = pml_req;
        ucx_req = (opal_common_ucx_request_t *)
                  ucp_tag_recv_nbx(ompi_pml_ucx.ucp_worker, buf,
                                   mca_pml_ucx_get_data_size(op_data, count),
                                   ucp_tag, ucp_tag_mask, &param);
        if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(ucx_req))) {
            PML_UCX_ERROR("ucx recv failed: %s",
                          ucs_status_string(UCS_PTR_STATUS(ucx_req)));
            PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &pml_req->ompi.super);
            return OMPI_ERROR;
        }
        pml_req->ompi.req_mpi_object.comm = comm;
        pml_req->ucx_req = ucx_req;
        /* With nbx + user_data, the callback always has the pml_req pointer even
         * for immediate completions, so no post-call test is needed. */
    }
#else
    ucx_req = (opal_common_ucx_request_t *)
              ucp_tag_recv_nb(ompi_pml_ucx.ucp_worker, buf, count,
                              mca_pml_ucx_get_datatype(datatype),
                              ucp_tag, ucp_tag_mask,
                              mca_pml_ucx_recv_completion);
    rc = mca_pml_ucx_post_recv(ucx_req, pml_req, comm);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != rc)) {
        return rc;
    }
#endif

    *request = &pml_req->ompi;
    return OMPI_SUCCESS;
}

int mca_pml_ucx_recv(void *buf, size_t count, ompi_datatype_t *datatype, int src,
                     int tag, struct ompi_communicator_t* comm,
                     ompi_status_public_t* mpi_status)
{
    /* coverity[bad_alloc_arithmetic] */
    void *req = PML_UCX_REQ_ALLOCA();
#if HAVE_DECL_UCP_TAG_RECV_NBX
    pml_ucx_datatype_t *op_data     = mca_pml_ucx_get_op_data(datatype);
    ucp_request_param_t *recv_param = &op_data->op_param.recv;
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_REQUEST |
                         (recv_param->op_attr_mask & UCP_OP_ATTR_FIELD_DATATYPE);
    param.datatype     = recv_param->datatype;
    param.request      = req;
#endif
    ucp_tag_t ucp_tag, ucp_tag_mask;
    ucp_tag_recv_info_t info;
    ucs_status_t status;
    int result;

    PML_UCX_TRACE_RECV("%s", buf, count, datatype, src, tag, comm, "recv");

    PML_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src, comm);
#if HAVE_DECL_UCP_TAG_RECV_NBX
    ucp_tag_recv_nbx(ompi_pml_ucx.ucp_worker, buf,
                     mca_pml_ucx_get_data_size(op_data, count),
                     ucp_tag, ucp_tag_mask, &param);
#else
    ucp_tag_recv_nbr(ompi_pml_ucx.ucp_worker, buf, count,
                     mca_pml_ucx_get_datatype(datatype),
                     ucp_tag, ucp_tag_mask, req);
#endif
    MCA_COMMON_UCX_PROGRESS_LOOP(ompi_pml_ucx.ucp_worker) {
        status = ucp_request_test(req, &info);
        if (status != UCS_INPROGRESS) {
            result = mca_pml_ucx_set_recv_status_public(mpi_status, status, &info);

#if SPC_ENABLE == 1
            size_t dt_size;
            ompi_datatype_type_size(datatype, &dt_size);
            SPC_USER_OR_MPI(tag, dt_size*count,
                            OMPI_SPC_BYTES_RECEIVED_USER, OMPI_SPC_BYTES_RECEIVED_MPI);
#endif
            return result;
        }
    }
}

__opal_attribute_always_inline__
static inline const char *mca_pml_ucx_send_mode_name(mca_pml_base_send_mode_t mode)
{
    switch (mode) {
    case MCA_PML_BASE_SEND_SYNCHRONOUS:
        return "sync";
    case MCA_PML_BASE_SEND_COMPLETE:
        return "complete";
    case MCA_PML_BASE_SEND_BUFFERED:
        return "buffered";
    case MCA_PML_BASE_SEND_READY:
        return "ready";
    case MCA_PML_BASE_SEND_STANDARD:
        return "standard";
    case MCA_PML_BASE_SEND_SIZE:
        return "size";
    default:
        return "unknown";
    }
}

int mca_pml_ucx_isend_init(const void *buf, size_t count, ompi_datatype_t *datatype,
                           int dst, int tag, mca_pml_base_send_mode_t mode,
                           struct ompi_communicator_t* comm,
                           struct ompi_request_t **request)
{
    mca_pml_ucx_persistent_request_t *req;
    ucp_ep_h ep;

    req = (mca_pml_ucx_persistent_request_t *)PML_UCX_FREELIST_GET(&ompi_pml_ucx.persistent_reqs);
    if (req == NULL) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    PML_UCX_TRACE_SEND("isend_init request *%p=%p", buf, count, datatype, dst,
                       tag, mode, comm, (void*)request, (void*)req)

    ep = mca_pml_ucx_get_ep(comm, dst);
    if (OPAL_UNLIKELY(NULL == ep)) {
        return OMPI_ERROR;
    }

    req->ompi.req_state           = OMPI_REQUEST_INACTIVE;
    req->ompi.req_mpi_object.comm = comm;
    req->flags                    = MCA_PML_UCX_REQUEST_FLAG_SEND;
    req->buffer                   = (void *)buf;
    req->count                    = count;
    req->tag                      = PML_UCX_MAKE_SEND_TAG(tag, comm);
    req->send.mode                = mode;
    req->send.ep                  = ep;
    req->ompi_datatype            = datatype;
    OMPI_DATATYPE_RETAIN(datatype);

    if (MCA_PML_BASE_SEND_BUFFERED == mode) {
        req->datatype = (ucp_datatype_t)NULL;
    } else {
        req->datatype = mca_pml_ucx_get_datatype(datatype);
    }

    *request = &req->ompi;
    return OMPI_SUCCESS;
}

static ucs_status_ptr_t
mca_pml_ucx_bsend(ucp_ep_h ep, const void *buf, size_t count,
                  ompi_datatype_t *datatype, uint64_t pml_tag)
{
    opal_common_ucx_request_t *ucx_req;
    mca_pml_ucx_req_t *pml_req;
    void *packed_data;
    size_t packed_length;
    size_t offset;
    uint32_t iov_count;
    struct iovec iov;
    opal_convertor_t opal_conv;

    OBJ_CONSTRUCT(&opal_conv, opal_convertor_t);
    opal_convertor_copy_and_prepare_for_send(ompi_proc_local_proc->super.proc_convertor,
                                             &datatype->super, count, buf, 0,
                                             &opal_conv);
    opal_convertor_get_packed_size(&opal_conv, &packed_length);

    packed_data = mca_pml_base_bsend_request_alloc_buf(packed_length);
    if (OPAL_UNLIKELY(NULL == packed_data)) {
        OBJ_DESTRUCT(&opal_conv);
        PML_UCX_ERROR("bsend: failed to allocate buffer");
        return UCS_STATUS_PTR(OMPI_ERROR);
    }

    iov_count    = 1;
    iov.iov_base = packed_data;
    iov.iov_len  = packed_length;

    PML_UCX_VERBOSE(8, "bsend of packed buffer %p len %zu", packed_data, packed_length);
    offset = 0;
    opal_convertor_set_position(&opal_conv, &offset);
    if (0 > opal_convertor_pack(&opal_conv, &iov, &iov_count, &packed_length)) {
        mca_pml_base_bsend_request_free(packed_data);
        OBJ_DESTRUCT(&opal_conv);
        PML_UCX_ERROR("bsend: failed to pack user datatype");
        return UCS_STATUS_PTR(OMPI_ERROR);
    }

    OBJ_DESTRUCT(&opal_conv);

    ucx_req = (opal_common_ucx_request_t *)
              ucp_tag_send_nb(ep, packed_data, packed_length,
                              ucp_dt_make_contig(1), pml_tag,
                              mca_pml_ucx_bsend_completion);
    if (NULL == ucx_req) {
        /* Completed immediately — callback freed ucx_req; free packed_data here. */
        mca_pml_base_bsend_request_free(packed_data);
        return NULL;
    }

    if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(ucx_req))) {
        mca_pml_base_bsend_request_free(packed_data);
        PML_UCX_ERROR("ucx bsend failed: %s", ucs_status_string(UCS_PTR_STATUS(ucx_req)));
        return UCS_STATUS_PTR(OMPI_ERROR);
    }

    /* In-progress: allocate a freelist slot to track packed_data until the
     * send completes and the completion callback can free it. */
    pml_req = (mca_pml_ucx_req_t *)PML_UCX_FREELIST_GET(&ompi_pml_ucx.reqs);
    if (OPAL_UNLIKELY(NULL == pml_req)) {
        /* Cannot track completion; cancel the send and bail. */
        ucp_request_cancel(ompi_pml_ucx.ucp_worker, ucx_req);
        mca_pml_base_bsend_request_free(packed_data);
        PML_UCX_ERROR("bsend: out of request slots");
        return UCS_STATUS_PTR(OMPI_ERROR);
    }
    pml_req->ompi.req_complete_cb_data = packed_data;
    pml_req->ucx_req                   = ucx_req;
    ucx_req->ext_req                   = pml_req;
    return NULL;
}

__opal_attribute_always_inline__
static inline ucs_status_ptr_t mca_pml_ucx_common_send(ucp_ep_h ep, const void *buf,
                                                       size_t count,
                                                       ompi_datatype_t *datatype,
                                                       ucp_datatype_t ucx_datatype,
                                                       ucp_tag_t tag,
                                                       mca_pml_base_send_mode_t mode,
                                                       ucp_send_callback_t cb)
{
    if (OPAL_UNLIKELY(MCA_PML_BASE_SEND_BUFFERED == mode)) {
        return mca_pml_ucx_bsend(ep, buf, count, datatype, tag);
    } else if (OPAL_UNLIKELY(MCA_PML_BASE_SEND_SYNCHRONOUS == mode)) {
        return ucp_tag_send_sync_nb(ep, buf, count, ucx_datatype, tag, cb);
    } else {
        return ucp_tag_send_nb(ep, buf, count, ucx_datatype, tag, cb);
    }
}

#if HAVE_DECL_UCP_TAG_SEND_NBX
__opal_attribute_always_inline__
static inline ucs_status_ptr_t
mca_pml_ucx_common_send_nbx(ucp_ep_h ep, const void *buf,
                            size_t count,
                            ompi_datatype_t *datatype,
                            ucp_tag_t tag,
                            mca_pml_base_send_mode_t mode,
                            ucp_request_param_t *param)
{
    pml_ucx_datatype_t *op_data = mca_pml_ucx_get_op_data(datatype);

    if (OPAL_UNLIKELY(MCA_PML_BASE_SEND_BUFFERED == mode)) {
        return mca_pml_ucx_bsend(ep, buf, count, datatype, tag);
    } else if (OPAL_UNLIKELY(MCA_PML_BASE_SEND_SYNCHRONOUS == mode)) {
        return ucp_tag_send_sync_nb(ep, buf, count,
                                    mca_pml_ucx_get_datatype(datatype), tag,
                                    (ucp_send_callback_t)param->cb.send);
    } else {
        return ucp_tag_send_nbx(ep, buf,
                                mca_pml_ucx_get_data_size(op_data, count),
                                tag, param);
    }
}
#endif

int mca_pml_ucx_isend(const void *buf, size_t count, ompi_datatype_t *datatype,
                      int dst, int tag, mca_pml_base_send_mode_t mode,
                      struct ompi_communicator_t* comm,
                      struct ompi_request_t **request)
{
    opal_common_ucx_request_t *ucx_req;
    mca_pml_ucx_req_t *pml_req;
    ucp_ep_h ep;
    ucp_tag_t pml_tag = PML_UCX_MAKE_SEND_TAG(tag, comm);

    PML_UCX_TRACE_SEND("i%ssend request *%p",
                       buf, count, datatype, dst, tag, mode, comm,
                       mode == MCA_PML_BASE_SEND_BUFFERED ? "b" : "",
                       (void*)request)

    ep = mca_pml_ucx_get_ep(comm, dst);
    if (OPAL_UNLIKELY(NULL == ep)) {
        return OMPI_ERROR;
    }

    /* Buffered send: mca_pml_ucx_bsend() manages its own freelist entry and
     * always returns to the caller as if the send completed. */
    if (OPAL_UNLIKELY(MCA_PML_BASE_SEND_BUFFERED == mode)) {
        ucx_req = (opal_common_ucx_request_t *)
                  mca_pml_ucx_bsend(ep, buf, count, datatype, pml_tag);
        if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(ucx_req))) {
            return OMPI_ERROR;
        }
        *request = &ompi_pml_ucx.completed_send_req;
        return OMPI_SUCCESS;
    }

    /* Allocate the ompi_request_t that will be returned to the user. */
    pml_req = (mca_pml_ucx_req_t *)PML_UCX_FREELIST_GET(&ompi_pml_ucx.reqs);
    if (OPAL_UNLIKELY(NULL == pml_req)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    mca_pml_ucx_request_reset(&pml_req->ompi);
    pml_req->ompi.req_mpi_object.comm = comm;
    pml_req->ucx_req                  = NULL;
    pml_req->detached                 = false;

#if HAVE_DECL_UCP_TAG_SEND_NBX
    {
        pml_ucx_datatype_t *op_data = mca_pml_ucx_get_op_data(datatype);
        if (OPAL_UNLIKELY(MCA_PML_BASE_SEND_SYNCHRONOUS == mode)) {
            /* ucp_tag_send_sync_nb has no nbx variant; fall back to nb. */
            ucx_req = (opal_common_ucx_request_t *)
                      ucp_tag_send_sync_nb(ep, buf, count,
                                           mca_pml_ucx_get_datatype(datatype),
                                           pml_tag, mca_pml_ucx_send_completion);
        } else {
            ucp_request_param_t param = op_data->op_param.isend;
            param.op_attr_mask |= UCP_OP_ATTR_FIELD_USER_DATA;
            param.user_data     = pml_req;
            ucx_req = (opal_common_ucx_request_t *)
                      ucp_tag_send_nbx(ep, buf,
                                       mca_pml_ucx_get_data_size(op_data, count),
                                       pml_tag, &param);
        }
    }
#else
    ucx_req = (opal_common_ucx_request_t *)
              mca_pml_ucx_common_send(ep, buf, count, datatype,
                                      mca_pml_ucx_get_datatype(datatype),
                                      pml_tag, mode,
                                      mca_pml_ucx_send_completion);
#endif

#if SPC_ENABLE == 1
    {
        size_t dt_size;
        ompi_datatype_type_size(datatype, &dt_size);
        SPC_USER_OR_MPI(tag, dt_size*count,
                        OMPI_SPC_BYTES_SENT_USER, OMPI_SPC_BYTES_SENT_MPI);
    }
#endif

    if (NULL == ucx_req) {
        /* Completed immediately. */
        PML_UCX_VERBOSE(8, "send completed immediately, returning completed request");
        PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &pml_req->ompi.super);
        *request = &ompi_pml_ucx.completed_send_req;
        return OMPI_SUCCESS;
    } else if (!UCS_PTR_IS_ERR(ucx_req)) {
        PML_UCX_VERBOSE(8, "got pml request %p ucx_req %p",
                        (void*)pml_req, (void*)ucx_req);
#if !HAVE_DECL_UCP_TAG_SEND_NBX
        /* For nb path (and sync_nb), link via ext_req. For nbx, user_data
         * already provides the link. */
        ucx_req->ext_req = pml_req;
#elif defined(HAVE_DECL_UCP_TAG_SEND_NBX)
        /* Sync sends use nb-style ext_req even when nbx is available. */
        if (OPAL_UNLIKELY(MCA_PML_BASE_SEND_SYNCHRONOUS == mode)) {
            ucx_req->ext_req = pml_req;
        }
#endif
        pml_req->ucx_req = ucx_req;
#if MPI_VERSION >= 4
        if (OPAL_LIKELY(mca_pml_ucx_request_cancel == pml_req->ompi.req_cancel)) {
            pml_req->ompi.req_cancel = mca_pml_ucx_request_cancel_send;
        }
#endif
        *request = &pml_req->ompi;
        return OMPI_SUCCESS;
    } else {
        PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &pml_req->ompi.super);
        PML_UCX_ERROR("ucx send failed: %s", ucs_status_string(UCS_PTR_STATUS(ucx_req)));
        return OMPI_ERROR;
    }
}

static inline __opal_attribute_always_inline__ int
mca_pml_ucx_send_nb(ucp_ep_h ep, const void *buf, size_t count,
                    ompi_datatype_t *datatype, ucp_datatype_t ucx_datatype,
                    ucp_tag_t tag, mca_pml_base_send_mode_t mode)
{
    opal_common_ucx_request_t *req;

    req = (opal_common_ucx_request_t *)mca_pml_ucx_common_send(ep, buf, count, datatype,
                                                               mca_pml_ucx_get_datatype(datatype),
                                                               tag, mode,
                                                               mca_pml_ucx_send_completion_empty);
    if (OPAL_LIKELY(req == NULL)) {
        return OMPI_SUCCESS;
    } else if (!UCS_PTR_IS_ERR(req)) {
        PML_UCX_VERBOSE(8, "got request %p", (void*)req);
        MCA_COMMON_UCX_WAIT_LOOP(req, ompi_pml_ucx.ucp_worker, "ucx send", ucp_request_free(req));
    } else {
        PML_UCX_ERROR("ucx send failed: %s", ucs_status_string(UCS_PTR_STATUS(req)));
        return OMPI_ERROR;
    }
}

#if HAVE_DECL_UCP_TAG_SEND_NBR
static inline __opal_attribute_always_inline__ int
mca_pml_ucx_send_nbr(ucp_ep_h ep, const void *buf, size_t count,
                     ompi_datatype_t *datatype, ucp_tag_t tag)
{
    /* coverity[bad_alloc_arithmetic] */
    ucs_status_ptr_t req = PML_UCX_REQ_ALLOCA();
#if HAVE_DECL_UCP_TAG_SEND_NBX
    pml_ucx_datatype_t *op_data = mca_pml_ucx_get_op_data(datatype);
    ucp_request_param_t param   = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_REQUEST |
                        (op_data->op_param.send.op_attr_mask & UCP_OP_ATTR_FIELD_DATATYPE) |
                        UCP_OP_ATTR_FLAG_FAST_CMPL,
        .datatype     = op_data->op_param.send.datatype,
        .request      = req
    };

    req = ucp_tag_send_nbx(ep, buf,
                           mca_pml_ucx_get_data_size(op_data, count),
                           tag, &param);
    if (OPAL_LIKELY(req == UCS_OK)) {
        return OMPI_SUCCESS;
    } else if (UCS_PTR_IS_ERR(req)) {
        PML_UCX_ERROR("%s failed: %d, %s", __func__, UCS_PTR_STATUS(req),
                      ucs_status_string(UCS_PTR_STATUS(req)));
        return OPAL_ERROR;
    }
#else
    ucs_status_t status;
    status = ucp_tag_send_nbr(ep, buf, count,
                              mca_pml_ucx_get_datatype(datatype), tag, req);
    if (OPAL_LIKELY(status == UCS_OK)) {
        return OMPI_SUCCESS;
    }
#endif

    MCA_COMMON_UCX_WAIT_LOOP(req, ompi_pml_ucx.ucp_worker, "ucx send nbr", (void)0);
}
#endif

int mca_pml_ucx_send(const void *buf, size_t count, ompi_datatype_t *datatype, int dst,
                     int tag, mca_pml_base_send_mode_t mode,
                     struct ompi_communicator_t* comm)
{
    ucp_ep_h ep;

    PML_UCX_TRACE_SEND("%s", buf, count, datatype, dst, tag, mode, comm,
                       mode == MCA_PML_BASE_SEND_BUFFERED ? "bsend" : "send");

    ep = mca_pml_ucx_get_ep(comm, dst);
    if (OPAL_UNLIKELY(NULL == ep)) {
        return OMPI_ERROR;
    }

#if SPC_ENABLE == 1
    size_t dt_size;
    ompi_datatype_type_size(datatype, &dt_size);
    SPC_USER_OR_MPI(tag, dt_size*count,
                    OMPI_SPC_BYTES_SENT_USER, OMPI_SPC_BYTES_SENT_MPI);
#endif

#if HAVE_DECL_UCP_TAG_SEND_NBR
    if (OPAL_LIKELY((MCA_PML_BASE_SEND_BUFFERED != mode) &&
                    (MCA_PML_BASE_SEND_SYNCHRONOUS != mode))) {
        return mca_pml_ucx_send_nbr(ep, buf, count, datatype,
                                    PML_UCX_MAKE_SEND_TAG(tag, comm));
    }
#endif

    return mca_pml_ucx_send_nb(ep, buf, count, datatype,
                               mca_pml_ucx_get_datatype(datatype),
                               PML_UCX_MAKE_SEND_TAG(tag, comm), mode);
}

int mca_pml_ucx_iprobe(int src, int tag, struct ompi_communicator_t* comm,
                       int *matched, ompi_status_public_t* mpi_status)
{
    static unsigned progress_count = 0;

    ucp_tag_t ucp_tag, ucp_tag_mask;
    ucp_tag_recv_info_t info;
    ucp_tag_message_h ucp_msg;

    PML_UCX_TRACE_PROBE("iprobe", src, tag, comm);

    PML_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src, comm);
    ucp_msg = ucp_tag_probe_nb(ompi_pml_ucx.ucp_worker, ucp_tag, ucp_tag_mask,
                               0, &info);
    if (ucp_msg != NULL) {
        *matched = 1;
        mca_pml_ucx_set_recv_status_public(mpi_status, UCS_OK, &info);
    } else  {
        (++progress_count % opal_common_ucx.progress_iterations) ?
            (int)ucp_worker_progress(ompi_pml_ucx.ucp_worker) : opal_progress();
        *matched = 0;
    }
    return OMPI_SUCCESS;
}

int mca_pml_ucx_probe(int src, int tag, struct ompi_communicator_t* comm,
                      ompi_status_public_t* mpi_status)
{
    ucp_tag_t ucp_tag, ucp_tag_mask;
    ucp_tag_recv_info_t info;
    ucp_tag_message_h ucp_msg;

    PML_UCX_TRACE_PROBE("probe", src, tag, comm);

    PML_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src, comm);

    MCA_COMMON_UCX_PROGRESS_LOOP(ompi_pml_ucx.ucp_worker) {
        ucp_msg = ucp_tag_probe_nb(ompi_pml_ucx.ucp_worker, ucp_tag,
                                   ucp_tag_mask, 0, &info);
        if (ucp_msg != NULL) {
            mca_pml_ucx_set_recv_status_public(mpi_status, UCS_OK, &info);
            return OMPI_SUCCESS;
        }
    }
}

int mca_pml_ucx_improbe(int src, int tag, struct ompi_communicator_t* comm,
                        int *matched, struct ompi_message_t **message,
                        ompi_status_public_t* mpi_status)
{
    static unsigned progress_count = 0;

    ucp_tag_t ucp_tag, ucp_tag_mask;
    ucp_tag_recv_info_t info;
    ucp_tag_message_h ucp_msg;

    PML_UCX_TRACE_PROBE("improbe", src, tag, comm);

    PML_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src, comm);
    ucp_msg = ucp_tag_probe_nb(ompi_pml_ucx.ucp_worker, ucp_tag, ucp_tag_mask,
                               1, &info);
    if (ucp_msg != NULL) {
        PML_UCX_MESSAGE_NEW(comm, ucp_msg, &info, message);
        PML_UCX_VERBOSE(8, "got message %p (%p)", (void*)*message, (void*)ucp_msg);
        *matched         = 1;
        mca_pml_ucx_set_recv_status_public(mpi_status, UCS_OK, &info);
    } else  {
        (++progress_count % opal_common_ucx.progress_iterations) ?
            (int)ucp_worker_progress(ompi_pml_ucx.ucp_worker) : opal_progress();
        *matched = 0;
    }
    return OMPI_SUCCESS;
}

int mca_pml_ucx_mprobe(int src, int tag, struct ompi_communicator_t* comm,
                         struct ompi_message_t **message,
                         ompi_status_public_t* mpi_status)
{
    ucp_tag_t ucp_tag, ucp_tag_mask;
    ucp_tag_recv_info_t info;
    ucp_tag_message_h ucp_msg;

    PML_UCX_TRACE_PROBE("mprobe", src, tag, comm);

    PML_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src, comm);
    MCA_COMMON_UCX_PROGRESS_LOOP(ompi_pml_ucx.ucp_worker) {
        ucp_msg = ucp_tag_probe_nb(ompi_pml_ucx.ucp_worker, ucp_tag, ucp_tag_mask,
                                   1, &info);
        if (ucp_msg != NULL) {
            PML_UCX_MESSAGE_NEW(comm, ucp_msg, &info, message);
            PML_UCX_VERBOSE(8, "got message %p (%p)", (void*)*message, (void*)ucp_msg);
            mca_pml_ucx_set_recv_status_public(mpi_status, UCS_OK, &info);
            return OMPI_SUCCESS;
        }
    }
}

int mca_pml_ucx_imrecv(void *buf, size_t count, ompi_datatype_t *datatype,
                         struct ompi_message_t **message,
                         struct ompi_request_t **request)
{
    mca_pml_ucx_req_t *pml_req;
    opal_common_ucx_request_t *ucx_req;
    struct ompi_communicator_t *comm = (*message)->comm;
    int rc;

    PML_UCX_TRACE_MRECV("imrecv", buf, count, datatype, message);

    pml_req = (mca_pml_ucx_req_t *)PML_UCX_FREELIST_GET(&ompi_pml_ucx.reqs);
    if (OPAL_UNLIKELY(NULL == pml_req)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    mca_pml_ucx_request_reset(&pml_req->ompi);
    pml_req->ucx_req  = NULL;
    pml_req->detached = false;

    ucx_req = (opal_common_ucx_request_t *)
              ucp_tag_msg_recv_nb(ompi_pml_ucx.ucp_worker, buf, count,
                                  mca_pml_ucx_get_datatype(datatype),
                                  (*message)->req_ptr,
                                  mca_pml_ucx_recv_completion);
    rc = mca_pml_ucx_post_recv(ucx_req, pml_req, comm);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != rc)) {
        return rc;
    }

    PML_UCX_MESSAGE_RELEASE(message);
    *request = &pml_req->ompi;
    return OMPI_SUCCESS;
}

int mca_pml_ucx_mrecv(void *buf, size_t count, ompi_datatype_t *datatype,
                        struct ompi_message_t **message,
                        ompi_status_public_t* status)
{
    mca_pml_ucx_req_t *pml_req;
    opal_common_ucx_request_t *ucx_req;
    ompi_request_t *req;
    struct ompi_communicator_t *comm = (*message)->comm;
    int rc;

    PML_UCX_TRACE_MRECV("mrecv", buf, count, datatype, message);

    pml_req = (mca_pml_ucx_req_t *)PML_UCX_FREELIST_GET(&ompi_pml_ucx.reqs);
    if (OPAL_UNLIKELY(NULL == pml_req)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    mca_pml_ucx_request_reset(&pml_req->ompi);
    pml_req->ucx_req  = NULL;
    pml_req->detached = false;

    ucx_req = (opal_common_ucx_request_t *)
              ucp_tag_msg_recv_nb(ompi_pml_ucx.ucp_worker, buf, count,
                                  mca_pml_ucx_get_datatype(datatype),
                                  (*message)->req_ptr,
                                  mca_pml_ucx_recv_completion);
    rc = mca_pml_ucx_post_recv(ucx_req, pml_req, comm);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != rc)) {
        return rc;
    }

    PML_UCX_MESSAGE_RELEASE(message);
    req = &pml_req->ompi;
    return ompi_request_wait(&req, status);
}

int mca_pml_ucx_start(size_t count, ompi_request_t** requests)
{
    mca_pml_ucx_persistent_request_t *preq;
    mca_pml_ucx_req_t *tmp_req;
    opal_common_ucx_request_t *ucx_req;
    size_t i;

    for (i = 0; i < count; ++i) {
        preq = (mca_pml_ucx_persistent_request_t *)requests[i];

        if ((preq == NULL) || (OMPI_REQUEST_PML != preq->ompi.req_type)) {
            /* Skip irrelevant requests */
            continue;
        }

        PML_UCX_ASSERT(preq->ompi.req_state != OMPI_REQUEST_INVALID);
        preq->ompi.req_state = OMPI_REQUEST_ACTIVE;
        mca_pml_ucx_request_reset(&preq->ompi);

        /* Allocate a non-persistent wrapper request to represent the in-flight
         * UCX operation.  The callback links back to preq via req_complete_cb_data. */
        tmp_req = (mca_pml_ucx_req_t *)PML_UCX_FREELIST_GET(&ompi_pml_ucx.reqs);
        if (OPAL_UNLIKELY(NULL == tmp_req)) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        mca_pml_ucx_request_reset(&tmp_req->ompi);
        tmp_req->ucx_req  = NULL;
        tmp_req->detached = false;
        tmp_req->ompi.req_complete_cb_data = preq;

        if (preq->flags & MCA_PML_UCX_REQUEST_FLAG_SEND) {
            ucx_req = (opal_common_ucx_request_t *)
                mca_pml_ucx_common_send(preq->send.ep,
                                        preq->buffer, preq->count,
                                        preq->ompi_datatype, preq->datatype,
                                        preq->tag, preq->send.mode,
                                        mca_pml_ucx_psend_completion);
            if (ucx_req == NULL) {
                /* Immediate success — no UCX request created */
                PML_UCX_VERBOSE(8, "send completed immediately, completing persistent "
                                "request %p", (void*)preq);
                mca_pml_ucx_request_reset(&tmp_req->ompi);
                PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &tmp_req->ompi.super);
                mca_pml_ucx_set_send_status(&preq->ompi.req_status, UCS_OK);
                ompi_request_complete(&preq->ompi, true);
            } else if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(ucx_req))) {
                PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &tmp_req->ompi.super);
                PML_UCX_ERROR("ucx send failed: %s",
                              ucs_status_string(UCS_PTR_STATUS(ucx_req)));
                return OMPI_ERROR;
            } else {
                PML_UCX_VERBOSE(8, "temporary request %p will complete persistent "
                                "request %p", (void*)tmp_req, (void*)preq);
                ucx_req->ext_req  = tmp_req;
                tmp_req->ucx_req  = ucx_req;
                preq->tmp_req     = &tmp_req->ompi;
            }
        } else {
            PML_UCX_VERBOSE(8, "start recv request %p", (void*)preq);
            ucx_req = (opal_common_ucx_request_t *)
                ucp_tag_recv_nb(ompi_pml_ucx.ucp_worker,
                                preq->buffer, preq->count,
                                preq->datatype, preq->tag,
                                preq->recv.tag_mask,
                                mca_pml_ucx_precv_completion);
            if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(ucx_req))) {
                PML_UCX_FREELIST_RETURN(&ompi_pml_ucx.reqs, &tmp_req->ompi.super);
                PML_UCX_ERROR("ucx recv failed: %s",
                              ucs_status_string(UCS_PTR_STATUS(ucx_req)));
                return OMPI_ERROR;
            }
            ucx_req->ext_req = tmp_req;
            tmp_req->ucx_req = ucx_req;
            preq->tmp_req    = &tmp_req->ompi;

            /* Detect immediate completion: the recv callback checks ext_req for
             * NULL and returns early if the message was already in the unexpected
             * queue when ucp_tag_recv_nb() was called.  Use ucp_request_test to
             * discover this case and complete preq manually. */
            {
                ucp_tag_recv_info_t info;
                ucs_status_t st = ucp_request_test(ucx_req, &info);
                if (UCS_INPROGRESS != st && !REQUEST_COMPLETE(&tmp_req->ompi)) {
                    /* Callback fired with NULL ext_req — complete now. */
                    PML_UCX_VERBOSE(8, "recv completed immediately, completing "
                                    "persistent request %p", (void*)preq);
                    mca_pml_ucx_set_recv_status(&tmp_req->ompi.req_status, st, &info);
                    tmp_req->ucx_req = NULL;
                    ucp_request_release(ucx_req);
                    mca_pml_ucx_persistent_request_complete(preq, &tmp_req->ompi);
                }
                /* else: callback already ran after ext_req was linked;
                 * mca_pml_ucx_preq_completion already completed preq and returned
                 * tmp_req to the freelist — nothing to do. */
            }
        }
    }

    return OMPI_SUCCESS;
}

int mca_pml_ucx_dump(struct ompi_communicator_t* comm, int verbose)
{
    return OMPI_SUCCESS;
}
