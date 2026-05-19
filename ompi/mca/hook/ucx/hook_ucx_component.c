/*
 * Copyright (c) 2026      NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "hook_ucx.h"

static int ompi_hook_ucx_component_open(void);
static int ompi_hook_ucx_component_close(void);
static int ompi_hook_ucx_component_register(void);


const ompi_hook_base_component_1_0_0_t mca_hook_ucx_component = {
    .hookm_version = {
        OMPI_HOOK_BASE_VERSION_1_0_0,

        .mca_component_name            = "ucx",
        .mca_component_major_version   = OMPI_MAJOR_VERSION,
        .mca_component_minor_version   = OMPI_MINOR_VERSION,
        .mca_component_release_version = OMPI_RELEASE_VERSION,
        .mca_open_component            = ompi_hook_ucx_component_open,
        .mca_close_component           = ompi_hook_ucx_component_close,
        .mca_query_component           = NULL,
        .mca_register_component_params = ompi_hook_ucx_component_register,
    },
    .hookm_data = {
        MCA_BASE_METADATA_PARAM_NONE
    },

    .hookm_mpi_init_top_post_opal      = ompi_hook_ucx_mpi_init_top_post_opal,
};

static int ompi_hook_ucx_component_open(void)
{
    opal_output_verbose(10, ompi_hook_base_framework.framework_output,
                        "hook/ucx: component_open()");
    return OMPI_SUCCESS;
}

static int ompi_hook_ucx_component_close(void)
{
    opal_output_verbose(10, ompi_hook_base_framework.framework_output,
                        "hook/ucx: component_close()");
    return OMPI_SUCCESS;
}

static int ompi_hook_ucx_component_register(void)
{
    return OMPI_SUCCESS;
}
