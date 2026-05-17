/*
 * Copyright (c) 2026      NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_HOOK_UCX_H
#define MCA_HOOK_UCX_H

#include "ompi_config.h"
#include "ompi/constants.h"
#include "ompi/mca/hook/hook.h"
#include "ompi/mca/hook/base/base.h"
#include "opal/util/output.h"

BEGIN_C_DECLS

OMPI_MODULE_DECLSPEC extern const ompi_hook_base_component_1_0_0_t mca_hook_ucx_component;

void ompi_hook_ucx_mpi_init_top_post_opal(int argc, char **argv,
                                          int requested, int *provided);

END_C_DECLS

#endif /* MCA_HOOK_UCX_H */
