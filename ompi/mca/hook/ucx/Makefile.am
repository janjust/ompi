#
# Copyright (c) 2026      NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AM_CPPFLAGS = $(hook_ucx_CPPFLAGS)

local_sources = \
	hook_ucx.h \
	hook_ucx_component.c \
	hook_ucx.c

if MCA_BUILD_ompi_hook_ucx_DSO
component_noinst =
component_install = mca_hook_ucx.la
else
component_noinst = libmca_hook_ucx.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_hook_ucx_la_SOURCES = $(local_sources)
mca_hook_ucx_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la $(hook_ucx_LIBS)
mca_hook_ucx_la_LDFLAGS = -module -avoid-version $(hook_ucx_LDFLAGS)

noinst_LTLIBRARIES = $(component_noinst)
libmca_hook_ucx_la_SOURCES = $(local_sources)
libmca_hook_ucx_la_LIBADD = $(hook_ucx_LIBS)
libmca_hook_ucx_la_LDFLAGS = -module -avoid-version $(hook_ucx_LDFLAGS)
