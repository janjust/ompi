/*
 * Copyright (c) 2026      NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "hook_ucx.h"
#include "ompi_config.h"

#include <ucm/api/ucm.h>


void ompi_hook_ucx_mpi_init_top_post_opal(int argc, char **argv,
                                          int requested, int *provided)
{
#if HAVE_DECL_UCM_TEST_EVENTS
    ucs_status_t status;

    status = ucm_test_events(UCM_EVENT_VM_MAPPED | UCM_EVENT_VM_UNMAPPED);
    opal_output_verbose(10, ompi_hook_base_framework.framework_output,
                        "hook/ucx: mpi_init_top_post_opal() ucm_test_events=%s",
                        ucs_status_string(status));
#endif
}
