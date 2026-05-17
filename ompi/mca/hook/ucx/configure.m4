#
# Copyright (c) 2026      NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AC_DEFUN([MCA_ompi_hook_ucx_CONFIG], [
    AC_CONFIG_FILES([ompi/mca/hook/ucx/Makefile])

    OMPI_CHECK_UCX([hook_ucx],
                   [hook_ucx_happy="yes"],
                   [hook_ucx_happy="no"])

    AS_IF([test "$hook_ucx_happy" = "yes"],
          [$1],
          [$2])

    AC_SUBST([hook_ucx_CPPFLAGS])
    AC_SUBST([hook_ucx_LDFLAGS])
    AC_SUBST([hook_ucx_LIBS])
])
