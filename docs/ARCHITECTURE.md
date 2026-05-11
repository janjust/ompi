# Open MPI Architecture

A developer-facing architectural reference for [Open MPI](https://github.com/open-mpi/ompi). The audience is contributors, integrators, and debuggers — people who need to *change* OMPI, not just call it.

This document complements but does not replace:

- [`HACKING.md`](../HACKING.md) — bootstrap & build instructions
- [`docs/contributing.rst`](contributing.rst) — contribution workflow
- [`docs/developers/`](developers/) — per-subsystem developer notes
- [`docs/getting-help.rst`](getting-help.rst) — user-facing help
- The Doxygen output of public headers and the rendered Sphinx site at <https://docs.open-mpi.org/>

If you came here to answer "how does the MCA framework actually work?", "where does PML selection happen?", "what does `coll/ucc` do at MPI_Init?", or "why is my custom component not being loaded?", the rest of this document is for you.

## Table of contents

1. [Charter & scope](#charter--scope)
2. [Top-level layout](#top-level-layout)
3. [The MCA framework](#the-mca-framework)
4. [Framework index — `ompi/mca/`](#framework-index--ompimca)
5. [Framework index — `opal/mca/`](#framework-index--opalmca)
6. [Framework index — `oshmem/mca/`](#framework-index--oshmemmca)
7. [Lifecycle hot paths](#lifecycle-hot-paths)
8. [Cross-library integration seams](#cross-library-integration-seams)
9. [MCA parameter system](#mca-parameter-system)
10. [Threading & progress](#threading--progress)
11. [Build system](#build-system)
12. [Configuration & runtime](#configuration--runtime)
13. [Debug & observability](#debug--observability)
14. [Top files for a debugger](#top-files-for-a-debugger)
15. [Interactions with sibling libraries](#interactions-with-sibling-libraries)
16. [Glossary](#glossary)

---

## Charter & scope

Open MPI is a high-performance, vendor-neutral, fully-featured implementation of the MPI standard (MPI 4.0+ at present), plus an OpenSHMEM implementation and a Modular Component Architecture (MCA) that lets every functional subsystem be a runtime-pluggable component.

The codebase is layered:

- **OPAL** (Open Portable Access Layer) — portability primitives, threading, memory, hardware abstractions, MCA core. *Not* MPI-aware.
- **OMPI** — the MPI layer. Communicators, datatypes, requests, collectives, point-to-point, RMA, MPI-IO.
- **OSHMEM** — OpenSHMEM layer (SPMD one-sided programming model).
- **3rd-party** — bundled PMIx (process manager interface), PRRTE (process runtime), hwloc, libevent, treematch.

OMPI's edge is the MCA: nearly every algorithm, transport, and policy is a swappable component selected at runtime. Modern HPC-X deployments resolve to roughly:

- `pml/ucx` for point-to-point messaging
- `osc/ucx` for one-sided / RMA
- `coll/ucc` (preferred) or `coll/hcoll` (legacy) for collectives, with `coll/tuned` and `coll/han` as software fallbacks
- `spml/ucx` for OpenSHMEM
- `accelerator/cuda` (or `rocm` / `ze`) for GPU buffer support

This document explains how those components plug in, when they're chosen, and where to look when something goes wrong.

---

## Top-level layout

```
ompi/
├── ompi/                     MPI layer
│   ├── mca/                  MPI-level frameworks (pml, coll, osc, mtl, io, fcoll, …)
│   ├── runtime/              ompi_mpi_init, ompi_mpi_finalize, hooks
│   ├── instance/             MPI 4.0 instance/session machinery
│   ├── communicator/         ompi_communicator_t lifecycle
│   ├── group/                ompi_group_t
│   ├── proc/                 ompi_proc_t (per-peer info)
│   ├── datatype/             MPI datatype engine
│   ├── op/                   reduction operators (MPI_OP)
│   ├── request/              ompi_request_t, generalised requests
│   ├── message/              MProbe/MMatched buffers
│   ├── attribute/            MPI attribute caching
│   ├── errhandler/           MPI error handlers
│   ├── info/                 MPI_Info
│   ├── file/                 MPI-IO file handle
│   ├── win/                  MPI_Win (one-sided window)
│   ├── peruse/               PERUSE performance instrumentation
│   ├── debuggers/            Debugger DLL interfaces (msgq, mpihandles)
│   ├── dpm/                  Dynamic process management
│   ├── interlib/             Inter-library cooperation
│   ├── mpi/                  Language bindings: c/, fortran/{use-mpi,use-mpi-f08,mpif-h}
│   ├── mpiext/               Extension framework
│   └── tools/ompi_info/      Runtime introspection tool
├── opal/                     Portability layer
│   ├── mca/                  Portable frameworks (btl, mpool, rcache, pmix, threads, …)
│   ├── runtime/              opal_init, opal_finalize, opal_progress
│   ├── util/                 output, show_help, error, argv, stacktrace, …
│   ├── threads/              thread primitives, atomics
│   ├── datatype/             low-level OPAL datatype engine
│   ├── class/                opal object system (refcount, OBJ_NEW/OBJ_RELEASE)
│   └── tools/                opal-level utilities
├── oshmem/                   OpenSHMEM layer
│   ├── mca/                  spml, scoll, atomic, memheap, sshmem
│   ├── shmem/                C bindings
│   └── runtime/              oshmem init/finalize
├── 3rd-party/                Bundled deps: openpmix/, prrte/, hwloc-*.tar.gz, libevent-*.tar.gz, treematch/
├── config/                   m4 macros: ompi_check_*, opal_*, oac_*
├── docs/                     This file, contributing.rst, developers/, features/, app-debug/
├── examples/                 Sample MPI programs
├── test/                     Internal tests
├── HACKING.md                Bootstrap instructions
├── autogen.pl                Build bootstrap (Perl, NOT autogen.sh)
└── configure.ac              Autoconf entry
```

The major subdirectories — `ompi/mca/`, `opal/mca/`, `oshmem/mca/` — are where most architectural decisions are encoded. The MCA framework is the organising abstraction.

---

## The MCA framework

OMPI's central concept. The MCA (Modular Component Architecture) lets each subsystem be:

- **A framework** — a category of plugin (e.g. `pml`, `coll`, `btl`).
- **One or more components** — implementations within a framework (e.g. `pml/ucx`, `pml/ob1`, `pml/cm`).
- **Modules** — per-instance state of a component (one per communicator for `coll`, one global for `pml`, etc.).

Frameworks live under `ompi/mca/<name>/`, `opal/mca/<name>/`, or `oshmem/mca/<name>/`. Each framework directory contains:

```
ompi/mca/<framework>/
├── <framework>.h            framework's public API
├── base/                    framework infrastructure (open/select/close, base helpers)
├── <component_a>/           one component
│   ├── configure.m4         build-time gating
│   ├── <framework>_<component>.{c,h}
│   ├── <framework>_<component>_component.c   the MCA component descriptor
│   └── …
├── <component_b>/
└── Makefile.am
```

### Canonical headers

| File | Role |
|------|------|
| [`opal/mca/mca.h`](../opal/mca/mca.h) | Component struct ([`:282`](../opal/mca/mca.h#L282)), version macros ([`:386`](../opal/mca/mca.h#L386)), component flags. |
| [`opal/mca/base/mca_base_framework.h`](../opal/mca/base/mca_base_framework.h) | Framework struct ([`:128-161`](../opal/mca/base/mca_base_framework.h#L128-L161)), lifecycle fns, `MCA_BASE_FRAMEWORK_DECLARE` macro ([`:267`](../opal/mca/base/mca_base_framework.h#L267)). |
| [`opal/mca/base/base.h`](../opal/mca/base/base.h) | Component-find/sort/select API, verbosity constants ([`:81-100`](../opal/mca/base/base.h#L81-L100)). |
| [`opal/mca/base/mca_base_var.h`](../opal/mca/base/mca_base_var.h) | MCA parameter system. |
| [`opal/mca/base/mca_base_pvar.h`](../opal/mca/base/mca_base_pvar.h) | MPI_T performance variables. |

### Component descriptor

Defined at [`opal/mca/mca.h:282-344`](../opal/mca/mca.h#L282-L344):

```c
struct mca_base_component_2_1_0_t {
    int mca_major_version, mca_minor_version, mca_release_version;
    char mca_project_name[…]; int mca_project_major/minor/release_version;
    char mca_type_name[…];     int mca_type_major/minor/release_version;
    char mca_component_name[…]; int mca_component_major/minor/release_version;

    mca_base_open_component_fn_t          mca_open_component;
    mca_base_close_component_fn_t         mca_close_component;
    mca_base_query_component_fn_t         mca_query_component;       /* deprecated */
    mca_base_register_component_params_fn_t mca_register_component_params;

    int mca_component_flags;       /* e.g. MCA_BASE_COMPONENT_FLAG_REQUIRED */
    char mca_component_path[…];
    void *mca_component_repository_item;
};
typedef struct mca_base_component_2_1_0_t mca_base_component_t;     /* :342 */
```

Frameworks subclass this struct: `mca_pml_base_component_2_1_0_t`, `mca_coll_base_component_3_1_0_t`, etc., each adding framework-specific fn pointers (init, query for module). The first member is always the base component.

### Framework descriptor

Defined at [`opal/mca/base/mca_base_framework.h:128-161`](../opal/mca/base/mca_base_framework.h#L128-L161):

```c
typedef struct mca_base_framework_t {
    char *framework_project;        /* "opal", "ompi", "oshmem" */
    char *framework_name;           /* "pml", "coll", "btl", … */
    char *framework_description;
    mca_base_framework_register_params_fn_t  framework_register;
    mca_base_framework_open_fn_t              framework_open;
    mca_base_framework_close_fn_t             framework_close;
    mca_base_framework_flags_t                framework_flags;
    int                                       framework_open_count;
    char                                     *framework_selection;
    int                                       framework_verbose;
    mca_base_open_only_dummy_component_t   *framework_static_components;
    opal_list_t                              framework_components;
    opal_list_t                              framework_failed_components;
    int                                       framework_refcnt;
    char                                     *framework_default_components_value;
} mca_base_framework_t;
```

Declared via the macro at [`mca_base_framework.h:267-281`](../opal/mca/base/mca_base_framework.h#L267-L281):

```c
MCA_BASE_FRAMEWORK_DECLARE(opal, foo, "description",
                           opal_foo_register, opal_foo_open, opal_foo_close,
                           opal_foo_static_components, MCA_BASE_FRAMEWORK_FLAG_LAZY)
```

This expands to a `mca_base_framework_t` named `opal_foo_base_framework`.

### Lifecycle: register → open → select → close

```
phase 1: register
    mca_base_framework_register(framework, flags)
        ├─ framework->framework_register(flags)
        └─ for each static + DSO component:
              component->mca_register_component_params()
              registers MCA params with the var registry

phase 2: open
    mca_base_framework_open(framework, flags)
        ├─ framework->framework_open()  (or default)
        ├─ mca_base_component_find — discover DSOs
        ├─ for each component:
        │     component->mca_open_component()
        │     OPAL_SUCCESS → add to framework_components list
        │     OPAL_ERR_NOT_AVAILABLE → silently skip
        │     other → add to framework_failed_components
        └─ framework_components is now populated

phase 3: select  (framework-specific; not all frameworks have an explicit select step)
    e.g. mca_pml_base_select() or mca_coll_base_comm_select(comm)
        ├─ for each opened component (sorted by priority):
        │     component->mca_*_component_init() returns module + priority
        ├─ pick highest-priority
        └─ install module's vtable

phase 4: close
    mca_base_framework_close(framework)
        ├─ framework->framework_close() (or default)
        └─ for each component: component->mca_close_component()
```

### Selection by priority

Components register a priority MCA param at registration time:

```c
ompi_pml_ucx.priority = 51;
mca_base_component_var_register(&mca_pml_ucx_component.pmlm_version, "priority",
                                "Priority of the UCX component",
                                MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                OPAL_INFO_LVL_3, MCA_BASE_VAR_SCOPE_LOCAL,
                                &ompi_pml_ucx.priority);
```

(See [`pml_ucx_component.c:57-63`](../ompi/mca/pml/ucx/pml_ucx_component.c#L57-L63).)

Components dynamically adjust their reported priority based on environment/hardware. `pml/ucx`'s component_init at [`pml_ucx_component.c:146`](../ompi/mca/pml/ucx/pml_ucx_component.c#L146) reports the registered priority *only* if UCX has device-level transport support; otherwise it returns a low priority and another PML wins.

User overrides:

```
OMPI_MCA_pml=ucx               # restrict to ucx
OMPI_MCA_pml=^ob1              # exclude ob1
OMPI_MCA_pml_ucx_priority=100  # raise priority
OMPI_MCA_coll=ucc,tuned,basic  # explicit fallback chain
```

### `configure.m4` per component

Each component has its own `configure.m4` declaring `OPAL_<PROJECT>_MCA_<framework>_<component>_SHOULD_BUILD`. Example: [`ompi/mca/pml/ucx/configure.m4`](../ompi/mca/pml/ucx/configure.m4) checks UCX presence via `OMPI_CHECK_UCX` and sets the component buildable.

---

## Framework index — `ompi/mca/`

MPI-layer frameworks. Listed alphabetically.

| Framework | Purpose | Notable components |
|-----------|---------|--------------------|
| `bml` | Byte Management Layer — sits between PML and BTL, manages BTL multiplexing. | `r2` (default) |
| [`coll`](../ompi/mca/coll/) | MPI collectives. Module per `MPI_Comm`. | `tuned` (algorithm-tuned), `basic` (always-works), `ucc` (UCC-backed), `han` (hierarchical), `adapt`, `acoll`, `accelerator` (GPU-aware), `inter`, `libnbc` (non-blocking), `monitoring`, `portals4`, `self`, `sync`, `xhc`, `ftagree`, `demo` |
| `common` | Shared helpers (not a real framework). | `monitoring`, `ompio`, `ubcl` |
| `fbtl` | File BTL — low-level file I/O for ROMIO/ompio. | `posix`, `ime` (DDN IME) |
| `fcoll` | File-collective operations. | `individual`, `dynamic`, `two_phase`, `vulcan` |
| `fs` | File system abstraction. | `posix`, `ufs` |
| `hook` | Hooks into MPI lifecycle. | `comm_method` |
| `io` | MPI-IO entry. | `romio341`, `ompio` |
| [`mtl`](../ompi/mca/mtl/) | Matching Transport Layer — alternative to PML+BTL for fabrics with HW matching. Used by `pml/cm`. | `ofi`, `psm2`, `portals4` |
| `op` | Reduction operators (MPI_OP backends). | `aarch64` (NEON), `avx` (AVX2/512), `base`, `accelerator` |
| [`osc`](../ompi/mca/osc/) | One-sided / RMA (`MPI_Win`). | `ucx` (UCX-backed), `rdma` (BTL-based), `sm` (shmem), `portals4`, `ubcl`, `monitoring` |
| `part` | Partitioned communication (MPI 4.0). | `persist` |
| [`pml`](../ompi/mca/pml/) | Point-to-point messaging layer. | `ucx` (priority 51 by default), `ob1` (BTL-based), `cm` (uses MTL), `monitoring`, `ubcl`, `v` |
| `sharedfp` | Shared file pointer. | `addproc`, `individual`, `lockedfile`, `sm` |
| `topo` | Communicator topology (Cartesian, graph). | `basic`, `treematch` |
| `vprotocol` | Virtual protocol wrapper for FT. | `pessimist` |

### `pml` deep dive

[`ompi/mca/pml/pml.h`](../ompi/mca/pml/pml.h) defines `mca_pml_base_module_t` — the vtable installed by the selected PML. Functions: `pml_send`, `pml_isend`, `pml_recv`, `pml_irecv`, `pml_iprobe`, `pml_probe`, `pml_iprobe`, `pml_dump`, `pml_add_procs`, `pml_del_procs`, `pml_enable`, `pml_progress` (rarely used; PMLs hook their own progress).

The shipped PMLs:

- **`pml/ucx`** — calls UCP directly. Wraps tag matching using `(comm_id, MPI_tag)` packed into the UCP tag. Fastest for fabrics with UCX support. Priority 51 by default. Files: [`pml_ucx.c`](../ompi/mca/pml/ucx/pml_ucx.c) (~1280 lines), [`pml_ucx.h`](../ompi/mca/pml/ucx/pml_ucx.h), [`pml_ucx_component.c`](../ompi/mca/pml/ucx/pml_ucx_component.c).
- **`pml/ob1`** — PML v4 RFC protocol over BTLs. Portable fallback when UCX unavailable. Multiplexes BTLs (e.g. `btl/tcp` + `btl/sm`).
- **`pml/cm`** — Convergence Mode. Thin wrapper over an MTL (e.g. `mtl/ofi`). Used when the fabric does its own matching.

### `coll` deep dive

[`ompi/mca/coll/coll.h`](../ompi/mca/coll/coll.h) defines `mca_coll_base_module_t` — a vtable per collective op (allreduce, bcast, gather, …). Each communicator gets its own module after `mca_coll_base_comm_select`.

The selection function lives at [`ompi/mca/coll/base/coll_base_comm_select.c:216`](../ompi/mca/coll/base/coll_base_comm_select.c#L216):

```c
int mca_coll_base_comm_select(ompi_communicator_t *comm)
{
    /* For each opened coll component:
     *   call component->collm_comm_query(comm, &priority)
     * Sort by priority. Install winners into comm->c_coll vtable.
     * Multiple components can co-exist; per-op fallbacks are possible.
     */
}
```

Notable coll components:

- **`coll/ucc`** — UCC-backed. Translates MPI datatype/op via [`coll_ucc_dtypes.h`](../ompi/mca/coll/ucc/coll_ucc_dtypes.h) (the `ompi_datatype_2_ucc_dt[]` table at [`:19-72`](../ompi/mca/coll/ucc/coll_ucc_dtypes.h#L19-L72) and `ompi_op_to_ucc_op_map[]` starting at [`:88`](../ompi/mca/coll/ucc/coll_ucc_dtypes.h#L88)). One UCC team per MPI communicator. Returns `OMPI_ERR_NOT_SUPPORTED` for collectives UCC can't handle (e.g. derived datatypes), triggering fallback.
- **`coll/tuned`** — Algorithm-selecting framework (knomial, recursive doubling, ring, Bruck, …) with per-`(coll, msg_size)` heuristics. The bread-and-butter SW path.
- **`coll/han`** — Hierarchical: decomposes by topology (intra-node + inter-node) and runs two-level collectives.
- **`coll/basic`** — Naive implementations using p2p. Always works. Last-resort fallback.
- **`coll/accelerator`** — GPU-aware wrappers that ensure the right `accelerator/*` ops are used for staging.
- **`coll/libnbc`** — Non-blocking collective implementations (MPI 3+).
- **`coll/sync`** — Inserts barriers periodically for better behaviour at scale (turn off with caution).
- **`coll/inter`** — Inter-communicator collectives.
- **`coll/self`** — Trivial size-1 communicator.
- **`coll/xhc`** — Cross-Host Collectives (specialised intra-node).
- **`coll/adapt`**, **`coll/acoll`** — Adaptive variants.

### `osc` deep dive

[`ompi/mca/osc/osc.h`](../ompi/mca/osc/osc.h) defines `ompi_osc_base_module_t` — the per-window vtable for `MPI_Put`, `MPI_Get`, `MPI_Accumulate`, fences, locks.

- **`osc/ucx`** — UCX-backed. Each window owns UCP endpoints to all peers and registers its memory via `ucp_mem_map` on creation; fence is `ucp_worker_flush`. See [`ompi/mca/osc/ucx/`](../ompi/mca/osc/ucx/).
- **`osc/rdma`** — Generic RDMA via BTL.
- **`osc/sm`** — Shared-memory window for intra-node.
- **`osc/portals4`** — Portals 4.0 transport.

---

## Framework index — `opal/mca/`

Portability-layer frameworks. The OPAL part of OMPI is *not* MPI-aware.

| Framework | Purpose | Notable components |
|-----------|---------|--------------------|
| `accelerator` | GPU/accelerator abstraction (managed-memory aware). | `cuda`, `rocm`, `ze`, `null` |
| `allocator` | Low-level memory allocators. | `basic`, `bucket` |
| `backtrace` | Crash backtrace. | `execinfo`, `printstack`, `none` |
| [`btl`](../opal/mca/btl/) | Byte Transfer Layer — point-to-point HW transport. | `tcp`, `sm` (shmem), `self`, `usnic`, `openib`, `ofi`, `ugni` |
| `common` | Shared helpers (not a real framework). | `ucx`, `sm`, `ofi`, `ubcl` |
| `dl` | Dynamic loader abstraction. | (built-in) |
| `hwloc` | Hardware topology (via hwloc lib). | `external` |
| `if` | Network interface enumeration. | `posix_ipv4`, `posix_ipv6`, `linux_ipv6` |
| `installdirs` | Install path lookup. | `config`, `env` |
| `memchecker` | Memory checker integration. | `valgrind` |
| `memcpy` | memcpy optimisations. | (built-in) |
| `memory` | Memory hook framework. | `patcher` (patches glibc), `none` |
| [`mpool`](../opal/mca/mpool/) | Memory pools (registration management). | `hugepage`, `lustre`, `memkind` |
| `patcher` | Runtime symbol patching. | `overwrite` |
| [`pmix`](../opal/mca/pmix/) | PMIx integration (process manager handoff). | `external`, `flux`, `s1`, `s2` |
| `rcache` | Registration cache. | `grdma` |
| `reachable` | Peer reachability discovery. | `netlink`, `weighted` |
| `shmem` | Shared-memory allocation. | `sysv`, `mmap`, `posix` |
| `smsc` | Shared Memory Single-Copy. | `cma` (CMA), `xpmem`, `knem`, `accelerator` |
| `threads` | Threading primitives. | `pthreads` (default), `argobots`, `qthreads` |
| `timer` | High-resolution timers. | platform-specific |

### `accelerator` framework

GPU support is centralised here. Components: `cuda`, `rocm`, `ze`, `null`. Each provides a vtable for: `mem_alloc`, `mem_free`, `memcpy_async`, `event_create/record/query`, `device_count`, `is_managed_memory`, `get_pointer_attributes`.

`opal_accelerator_*` ops are called from `coll/accelerator`, `pml/ucx` (for buffer staging when UCX path is HOST-only), `osc/ucx`, `op/accelerator` (for GPU-side reductions). Selection via priority; usually one accelerator wins on a given system.

### `btl` framework

Used only by `pml/ob1` and `osc/rdma`. Defines `mca_btl_base_module_t` with `btl_send`, `btl_put`, `btl_get`, `btl_atomic_op`, `btl_register`, `btl_progress`. Components include `btl/tcp`, `btl/sm` (shmem), `btl/self`, `btl/openib` (legacy IB), `btl/ofi` (libfabric), `btl/usnic`, `btl/ugni`.

In an HPC-X-on-IB deployment, BTLs are largely irrelevant — `pml/ucx` bypasses them and talks UCX directly. They matter only when `pml/ob1` is selected (e.g. `OMPI_MCA_pml=ob1`).

### `pmix` framework

PMIx is OMPI's process manager interface — the IPC for "ask the launcher who else is in this job, where they are, and how to reach them". OMPI uses PMIx for:

- Global rank assignment
- Modex (module exchange) — every process publishes its endpoint info, every process pulls peers' info
- Process-set membership (MPI 4.0 sessions)
- Notification (pre-emption, fault)

The `pmix/external` component is now the standard — links against the bundled or system PMIx (`3rd-party/openpmix/`). Older `s1`/`s2` were for SLURM PMI-1/PMI-2 emulation.

### `common/ucx`

[`opal/mca/common/ucx/`](../opal/mca/common/ucx/) — *not* a framework, but a "common" library shared by every OMPI component that touches UCX (`pml/ucx`, `osc/ucx`, `spml/ucx`). Exports:

- [`opal_common_ucx_module_t`](../opal/mca/common/ucx/common_ucx.h#L88) — global UCX state: `verbose`, `progress_iterations`, `tls`, `devices`, `opal_mem_hooks` flag.
- `opal_common_ucx_mca_register()` / `_deregister()` — register the shared MCA params (`opal_common_ucx_tls`, `_devices`, `_verbose`, etc.) and load UCX
- `opal_common_ucx_support_level(context)` — tri-state: `NONE`, `TRANSPORT` (UCX present but no compatible HCA found), `DEVICE` (good to go) — defined at [`common_ucx.h:103-113`](../opal/mca/common/ucx/common_ucx.h#L103-L113)
- Helper macros: [`MCA_COMMON_UCX_VERBOSE`](../opal/mca/common/ucx/common_ucx.h#L54), [`MCA_COMMON_UCX_PROGRESS_LOOP`](../opal/mca/common/ucx/common_ucx.h#L62), `MCA_COMMON_UCX_WAIT_LOOP`

PMLs/OSCs that use UCX call `opal_common_ucx_mca_var_register(&component->version)` to inherit the shared params and then check `opal_common_ucx_support_level()` to decide whether to bid for selection.

---

## Framework index — `oshmem/mca/`

OpenSHMEM-layer frameworks.

| Framework | Purpose | Notable components |
|-----------|---------|--------------------|
| [`spml`](../oshmem/mca/spml/) | Symmetric PML — the SHMEM one-sided primitive layer. | `ucx` (default for HPC-X) |
| `scoll` | SHMEM collectives. | `basic`, `mpi`, `ucc`, `fca` |
| `atomic` | SHMEM atomics. | `mxm` (legacy), `ucx` |
| `memheap` | Symmetric heap manager. | `ptmalloc`, `external` |
| `sshmem` | Symmetric shared memory backing. | `sysv`, `mmap`, `verbs` |

`spml/ucx` is the OSHMEM equivalent of `pml/ucx`. Same UCX context; different API surface (symmetric heap addressed by remote-rank + offset).

---

## Lifecycle hot paths

### `MPI_Init` — the biggest trace

Entry point: `ompi_mpi_init` at [`ompi/runtime/ompi_mpi_init.c:340`](../ompi/runtime/ompi_mpi_init.c#L340).

```
ompi_mpi_init(argc, argv, requested, provided, reinit_ok)
  │
  ├─ ompi_mpi_state CAS:  NOT_INITIALIZED → INIT_STARTED
  │   (atomic; second caller fails fast)
  │
  ├─ ompi_hook_base_mpi_init_top()         pre-init hooks
  │
  ├─ ompi_mpi_thread_level(req, provided)  thread level negotiation
  │     sets ompi_mpi_thread_multiple, ompi_mpi_main_thread
  │
  ├─ ompi_mpi_instance_init(ts_level, info, errhandler, &instance, argc, argv)
  │     // ompi/instance/instance.c:851
  │     │
  │     ├─ opal_init() — OPAL framework regs + opens
  │     │     opens: dl, threads, memory, timer, hwloc, accelerator,
  │     │            pmix, btl (if needed), mpool, rcache, smsc, allocator, …
  │     │
  │     ├─ PMIx connect — ompi_rte_init() handshake with launcher
  │     │     ranks discover each other; fence #1 (modex barrier)
  │     │
  │     ├─ opens MPI-layer frameworks in order:
  │     │     mca_pml_base_open()
  │     │     mca_bml_base_open()    (used by pml/ob1)
  │     │     mca_coll_base_open()
  │     │     mca_osc_base_open()
  │     │     mca_op_base_open()
  │     │     mca_io_base_open()
  │     │     …
  │     │
  │     ├─ mca_pml_base_select() — picks the PML
  │     │     for each pml component: query priority + module
  │     │     install winning module's vtable globally
  │     │
  │     ├─ ompi_proc_init() — populate ompi_proc_t array
  │     │
  │     ├─ MPI_COMM_WORLD construction:
  │     │     ompi_communicator_t allocated
  │     │     mca_coll_base_comm_select(MPI_COMM_WORLD)
  │     │       (per-coll vtable installed; coll/ucc creates UCC team here)
  │     │     mca_pml_base_add_procs() — PML adds endpoints to all peers
  │     │
  │     └─ ompi_mpi_state → INIT_COMPLETE
  │
  ├─ ompi_hook_base_mpi_init_bottom()       post-init hooks
  │
  └─ return MPI_SUCCESS
```

The framework open order is *critical* — `coll` cannot open before `pml`, because most coll components rely on the chosen PML's vtable. Reversed order in finalize is similarly load-bearing.

### `MPI_Finalize`

Entry point: `ompi_mpi_finalize` at [`ompi/runtime/ompi_mpi_finalize.c:109`](../ompi/runtime/ompi_mpi_finalize.c#L109).

```
ompi_mpi_finalize()
  │
  ├─ ompi_mpi_state CAS: INIT_COMPLETE → FINALIZE_STARTED
  │
  ├─ ompi_hook_base_mpi_finalize_top()
  │
  ├─ MPI_Barrier(MPI_COMM_WORLD)            "is everyone here?"
  │
  ├─ flush all in-flight requests
  │     ompi_request_persistent_noop_create_finalize(), …
  │
  ├─ tear down communicators in reverse creation order
  │     ompi_comm_finalize() — calls coll module finalize
  │
  ├─ close MPI-layer frameworks in REVERSE open order:
  │     mca_op_base_close()
  │     mca_io_base_close()
  │     mca_osc_base_close()
  │     mca_coll_base_close()
  │     mca_bml_base_close()
  │     mca_pml_base_close()
  │
  ├─ ompi_mpi_instance_finalize()
  │     PMIx fence #N (synchronise teardown)
  │     PMIx finalize
  │     opal_finalize() — OPAL framework closes in reverse
  │
  ├─ ompi_hook_base_mpi_finalize_bottom()
  │
  └─ ompi_mpi_state → FINALIZED
```

Bugs cluster here. Common patterns:

- A request leaked → finalize fence hangs because peer is waiting on a send that was never matched.
- A coll component fails to close cleanly because its child UCC/UCX teams were already torn down by a different fence.
- `accelerator/cuda` closes before `osc/ucx` releases its GPU memory → UCX gets a stale CUDA pointer and aborts.

### Communicator creation

[`ompi/communicator/comm_init.c`](../ompi/communicator/comm_init.c) (638 lines) and [`comm.c`](../ompi/communicator/comm.c) (~2750 lines).

```
ompi_comm_create / ompi_comm_split / ompi_comm_dup
  │
  ├─ ompi_comm_new() — allocate communicator + group
  ├─ assign communicator ID via PMIx-collective ompi_comm_nextcid (needs all peers)
  ├─ build group / rank table
  ├─ mca_coll_base_comm_select(comm)
  │     // ompi/mca/coll/base/coll_base_comm_select.c:216
  │     queries every coll component, picks per-op winners
  │     installs comm->c_coll vtable
  │
  └─ ompi_comm_activate() — make it usable
```

`comm->c_coll` is a struct of function pointers — `coll_allreduce`, `coll_bcast`, etc. — installed during `comm_select`. When MPI_Allreduce is called, it routes through `comm->c_coll->coll_allreduce(…)`.

### PML send/recv hot path

For `pml/ucx`, the modern API is `mca_pml_ucx_isend` at [`pml_ucx.c:899`](../ompi/mca/pml/ucx/pml_ucx.c#L899) and `mca_pml_ucx_irecv` at [`pml_ucx.c:636`](../ompi/mca/pml/ucx/pml_ucx.c#L636), with synchronous `mca_pml_ucx_send`/`_recv` at [`:1022`](../ompi/mca/pml/ucx/pml_ucx.c#L1022) / [`:673`](../ompi/mca/pml/ucx/pml_ucx.c#L673).

- Send path: builds the UCP tag from `(comm_id, MPI_tag)`, packs the datatype if non-contig, calls `ucp_tag_send_nb`/`_nbx` against the peer's `ucp_ep_h` (cached per `ompi_proc_t`).
- Recv path: posts via `ucp_tag_recv_nb`/`_nbx`/`_nbr` against the worker, with tag mask covering source rank wildcards. The returned UCS request pointer is type-punned to `ompi_request_t`.

### One-sided / RMA hot path

`osc/ucx` window creation: each rank does `ucp_mem_map` on its window buffer, then publishes the rkey via PMIx-modex. RMA ops (`MPI_Put`, `MPI_Get`, `MPI_Accumulate`) call `ucp_put_nbx` / `ucp_get_nbx` / `ucp_atomic_op_nbx`. Synchronisation modes:

- **Fence**: `ucp_worker_flush_nbx` on all eps + `MPI_Barrier`.
- **PSCW** (Post/Start/Complete/Wait): point-to-point synchronisation between origin and target ranks.
- **Lock**: passive target; uses UCX atomics for distributed locks.

---

## Cross-library integration seams

The OMPI ↔ UCX ↔ UCC ↔ HCOLL boundaries are where most HPC-X bugs live.

### `ompi/mca/pml/ucx/` — UCX-backed point-to-point

- Component: [`pml_ucx_component.c`](../ompi/mca/pml/ucx/pml_ucx_component.c) — `priority = 51` ([`:57`](../ompi/mca/pml/ucx/pml_ucx_component.c#L57)), reports actual priority based on `opal_common_ucx_support_level()` at [`pml_ucx_component.c:146`](../ompi/mca/pml/ucx/pml_ucx_component.c#L146).
- Hot path: [`pml_ucx.c`](../ompi/mca/pml/ucx/pml_ucx.c) — `isend`/`irecv`/`send`/`recv` wrappers around `ucp_tag_send_nbx`/`ucp_tag_recv_nbx`.
- Registers with the shared common UCX module (no per-component UCP context).

### `ompi/mca/osc/ucx/` — UCX-backed RMA

- [`osc_ucx.h`](../ompi/mca/osc/ucx/osc_ucx.h) defines the per-window module
- `osc_ucx_comm.c` — window creation, UCP memory mapping, rkey exchange
- `osc_ucx_active_target.c`, `osc_ucx_passive_target.c` — synchronisation modes

### `ompi/mca/coll/ucc/` — UCC-backed collectives

- [`coll_ucc.h`](../ompi/mca/coll/ucc/coll_ucc.h) — module struct, function-pointer table
- [`coll_ucc_component.c`](../ompi/mca/coll/ucc/coll_ucc_component.c) — `ucc_init_version` called once per OMPI process, `ucc_context_create` per CL, per-op enable flags
- [`coll_ucc_module.c`](../ompi/mca/coll/ucc/coll_ucc_module.c) — per-communicator module: `ucc_team_create_post` + `ucc_team_create_test` driven from `mca_coll_ucc_module_enable`
- [`coll_ucc_dtypes.h`](../ompi/mca/coll/ucc/coll_ucc_dtypes.h) — translation tables:
  - `ompi_datatype_2_ucc_dt[OPAL_DATATYPE_MAX_PREDEFINED]` ([`:19-72`](../ompi/mca/coll/ucc/coll_ucc_dtypes.h#L19-L72))
  - `ompi_op_to_ucc_op_map[OMPI_OP_BASE_FORTRAN_OP_MAX + 1]` ([`:88+`](../ompi/mca/coll/ucc/coll_ucc_dtypes.h#L88))
- Per-collective files: `coll_ucc_allreduce.c`, `coll_ucc_bcast.c`, `coll_ucc_alltoall.c`, etc. Each builds `ucc_coll_args_t`, calls `ucc_collective_init`, `ucc_collective_post`, and stages completion via `ucc_context_progress` driven by `opal_progress`.
- Fallback: when UCC returns `UCC_ERR_NOT_SUPPORTED` (unsupported datatype, asymmetric memory, etc.), the module returns `OMPI_ERR_NOT_SUPPORTED`, prompting the framework to dispatch to the next-priority coll component.

### `coll/hcoll` — HCOLL-backed collectives (vendor branches only)

The HCOLL collective offload library predates UCC. The corresponding `coll/hcoll` component is shipped on Mellanox / HPC-X vendor branches but does *not* exist on upstream `main`. New work uses UCC; HCOLL is mentioned here only because the seam table and HPC-X discussion reference it. If you need to debug HCOLL integration, switch to the appropriate vendor branch (e.g. `mellanox/v5.0.x_hpcx`).

### `oshmem/mca/spml/ucx/` — OSHMEM ↔ UCX

[`oshmem/mca/spml/ucx/`](../oshmem/mca/spml/ucx/) — symmetric one-sided over UCX. Key files: `spml_ucx.c`, `spml_ucx_component.c`. Same UCP context as PML/OSC; different addressing model (rank + symmetric offset vs. window).

### `opal/mca/common/ucx/` — shared UCX init

Already covered in [Framework index — `opal/mca/`](#framework-index--opalmca). Owns the global `opal_common_ucx_module_t`.

### Seam summary table

| Seam | OMPI side | UCX/UCC side |
|------|-----------|--------------|
| MPI p2p | [`ompi/mca/pml/ucx/pml_ucx.c`](../ompi/mca/pml/ucx/pml_ucx.c) | UCX `src/ucp/tag/{eager_snd,tag_send}.c` |
| MPI RMA | [`ompi/mca/osc/ucx/`](../ompi/mca/osc/ucx/) | UCX `src/ucp/rma/`, atomics |
| MPI coll → UCC | [`ompi/mca/coll/ucc/`](../ompi/mca/coll/ucc/) | UCC `src/core/ucc_coll.c`, `tl/ucp` → UCX `src/ucp/{tag,am,rma}/` |
| MPI coll → HCOLL (vendor branches only) | `ompi/mca/coll/hcoll/` (downstream) | HCOLL (external library) |
| UCX shared init | [`opal/mca/common/ucx/`](../opal/mca/common/ucx/) | UCX `src/ucp/api/ucp.h` |
| OSHMEM | [`oshmem/mca/spml/ucx/`](../oshmem/mca/spml/ucx/) | UCX `src/ucp/`, `src/uct/` |
| Datatype/op translation | [`ompi/mca/coll/ucc/coll_ucc_dtypes.h:19-114`](../ompi/mca/coll/ucc/coll_ucc_dtypes.h#L19-L114) | UCC `src/ucc/api/ucc.h` |

---

## MCA parameter system

Defined in [`opal/mca/base/mca_base_var.h`](../opal/mca/base/mca_base_var.h) (744 lines).

### Parameter declaration

```c
mca_base_component_var_register(
    /* component */ &mca_pml_ucx_component.pmlm_version,
    /* name      */ "priority",
    /* doc       */ "Priority of the UCX component",
    /* type      */ MCA_BASE_VAR_TYPE_INT,
    /* enums     */ NULL,
    /* flags     */ 0,
    /* arg       */ 0,
    /* level     */ OPAL_INFO_LVL_3,
    /* scope     */ MCA_BASE_VAR_SCOPE_LOCAL,
    /* storage   */ &ompi_pml_ucx.priority);
```

Types: `INT`, `UINT`, `LONG`, `ULONG`, `SIZE_T`, `BOOL`, `STRING`, `VERSION_STRING`, `DOUBLE`, `LONG_DOUBLE`. Enums: `mca_base_var_enum_t` constructed from a list of `(name, value)` pairs; the parser accepts symbolic values.

Scopes:

- `MCA_BASE_VAR_SCOPE_LOCAL` — process-local
- `MCA_BASE_VAR_SCOPE_READONLY` — set once; later changes ignored
- `MCA_BASE_VAR_SCOPE_CONSTANT` — compile-time
- `MCA_BASE_VAR_SCOPE_ALL_EQ` — must agree across all procs in a comm
- `MCA_BASE_VAR_SCOPE_GROUP` — must agree within a group

Levels (passed to `OPAL_INFO_LVL_*`): tuneable user (1) → tuneable detail (3) → developer (9). `ompi_info -a -l <level>` filters.

### Resolution order

```
command line  (--mca pml_ucx_priority 100)
   ↓
environment   (OMPI_MCA_pml_ucx_priority=100)
   ↓
config file   (~/.openmpi/mca-params.conf, $prefix/etc/openmpi-mca-params.conf)
   ↓
default
```

`PRTE_MCA_*` covers the runtime-environment side (PRRTE-managed).

### MPI_T pvar interface

[`opal/mca/base/mca_base_pvar.h`](../opal/mca/base/mca_base_pvar.h) (578 lines). Components register performance variables (counters, gauges) via `mca_base_pvar_register`. Tools query through `MPI_T_pvar_*` API. `coll/monitoring`, `pml/monitoring`, `osc/monitoring`, `common/monitoring` provide message-counting pvars.

---

## Threading & progress

### `opal_progress`

The single most important entry point in OMPI. [`opal/runtime/opal_progress.c:216`](../opal/runtime/opal_progress.c#L216):

```c
int opal_progress(void)
{
    /* call all registered callbacks
     * call event library (libevent) once
     * every 8th call, run low-priority callbacks
     */
}
```

Registered callbacks come from BTLs, PMLs, OSCs, PMIx — they call `opal_progress_register(fn)` on init. The most important consumer for HPC-X is `pml/ucx`, which registers a callback that calls `ucp_worker_progress(common_ucx_worker)`.

`opal_progress` is called from:

- `MPI_Test` / `MPI_Wait` polling loops
- Inside collectives that block on completion
- The async progress thread (if `--enable-mt` and an async mode is configured)

Tuning knobs:

```
OMPI_MCA_opal_event_include=epoll      # libevent backend
OMPI_MCA_opal_progress_event_users=…
```

### Thread modes

`MPI_Init_thread(MPI_THREAD_MULTIPLE)` triggers heavier locking throughout. `ompi_mpi_thread_multiple` is the global atomic flag (set at `ompi_mpi_init` thread-level negotiation). When enabled:

- Critical sections use `opal_mutex_t` (pthread_mutex_t)
- The opal `threads` framework determines the underlying primitives (`pthreads`, `qthreads`, `argobots`)
- `OPAL_THREAD_LOCK` / `OPAL_THREAD_UNLOCK` macros are no-ops in single-threaded builds

### Async progress thread

Optional. Enabled at build time with `--enable-mt` and at runtime via OPAL params. A dedicated thread calls `opal_progress` periodically so that overlap is achieved when the application doesn't explicitly call MPI_Test/MPI_Wait.

---

## Build system

### Bootstrap — `autogen.pl`, **not** `autogen.sh`

```
$ ./autogen.pl
```

This is a 1819-line Perl script ([`autogen.pl`](../autogen.pl)) that:

1. Discovers all frameworks under `<project>/mca/`
2. Discovers all components under each framework
3. Reads each component's `configure.m4`
4. Generates `config/autogen_found_items.m4` listing them
5. Runs `autoreconf -fiv`

If you add a framework or component, you must rerun `autogen.pl`.

### `configure.ac`

[`configure.ac`](../configure.ac) (1577 lines). High-level flow:

1. Project version macros (`AC_INIT`)
2. Compiler / OS / sysdep checks
3. Optional dependency detection — one m4 macro per major dep:
   - `OMPI_CHECK_UCX` — [`config/ompi_check_ucx.m4`](../config/ompi_check_ucx.m4) (142 lines), uses `OAC_CHECK_PACKAGE([ucx], …)`
   - `OMPI_CHECK_HCOLL`
   - `OMPI_CHECK_PMIX`
   - `OMPI_CHECK_PRRTE`
   - `OMPI_CHECK_HWLOC`
   - `OMPI_CHECK_LIBEVENT`
4. 3rd-party bundling (PMIx, PRRTE, hwloc, libevent — built as subdirs)
5. Per-framework component evaluation: every component's `configure.m4` is sourced, and the `OPAL_<PROJECT>_MCA_<framework>_<component>_SHOULD_BUILD` decision is made
6. `AC_CONFIG_FILES` — the long list of generated Makefiles

### `--with-X` flags

Common for HPC-X builds:

```bash
./configure \
    --prefix=/opt/openmpi \
    --with-ucx=/opt/ucx \
    --with-ucc=/opt/ucc \
    --with-hcoll=/opt/hcoll \
    --with-cuda=/usr/local/cuda \
    --with-pmix=internal \
    --with-prrte=internal \
    --with-libevent=internal \
    --with-hwloc=internal \
    --enable-mca-no-build=btl-uct,btl-openib    # exclude unwanted components
```

### 3rd-party bundling

[`3rd-party/`](../3rd-party/) ships PMIx, PRRTE, hwloc, libevent. The `--with-pmix=internal` form builds the bundled copy; `--with-pmix=external` (or `--with-pmix=/path`) uses a system install.

---

## Configuration & runtime

### Env-var convention

```
OMPI_MCA_<framework>_<component>_<param>=<value>
OMPI_MCA_<framework>=<comp>[,<comp>][,^<exclude>]    # selection
OMPI_MCA_<framework>_<param>=<value>                  # framework-level
PRTE_MCA_*                                            # PRRTE side
PMIX_MCA_*                                            # PMIx side
OPAL_MCA_*                                            # OPAL-only params
```

### Configuration files

Search path (each layered on top of the previous):

1. `$prefix/etc/openmpi-mca-params.conf`
2. `~/.openmpi/mca-params.conf`
3. `OMPI_MCA_PARAM_FILE` env var

Format:

```
pml = ucx
coll_ucc_priority = 50
btl_tcp_if_include = eth0
opal_common_ucx_tls = rc,sm,self
```

### `ompi_info` — discovery tool

[`ompi/tools/ompi_info/ompi_info.c`](../ompi/tools/ompi_info/ompi_info.c).

```
ompi_info -a                       # all params
ompi_info --param coll all         # all coll components' params
ompi_info --param pml ucx          # pml/ucx params only
ompi_info --param all all -l 9     # all params at developer level
ompi_info --all --parsable         # machine-readable
ompi_info -c                       # config (build-time options)
ompi_info --internal               # internal/private params
```

When you don't know what params exist, `ompi_info --param <fw> <comp>` is the answer.

### `mpirun` / `prun` flags worth knowing

```
--display-allocation              # show what nodes/slots PRRTE got
--display-map                     # show actual rank-to-node map
--report-bindings                 # core-binding summary
--mca <fw>_<comp>_<param> <val>   # MCA on the command line
-x VAR=val                        # forward env var to all ranks
```

---

## Debug & observability

### `opal_output` and verbosity

[`opal/util/output.h`](../opal/util/output.h) (592 lines). Every framework opens an output stream at init:

```c
mca_pml_base_output = opal_output_open(NULL);
opal_output_set_verbosity(mca_pml_base_output, mca_pml_base_verbose);
opal_output_verbose(level, mca_pml_base_output, "PML found %d components", n);
```

Verbosity is set by `OMPI_MCA_<framework>_base_verbose=<level>`. The level constants live at [`opal/mca/base/base.h:81-100`](../opal/mca/base/base.h#L81-L100):

| Constant | Value | Meaning |
|----------|-------|---------|
| `MCA_BASE_VERBOSE_NONE` | -1 | Silent |
| `MCA_BASE_VERBOSE_ERROR` | 0 | Errors only |
| `MCA_BASE_VERBOSE_COMPONENT` | 10 | Selection events |
| `MCA_BASE_VERBOSE_WARN` | 20 | Warnings |
| `MCA_BASE_VERBOSE_INFO` | 40 | Rationale, system probe results |
| `MCA_BASE_VERBOSE_TRACE` | 60 | Function trace |
| `MCA_BASE_VERBOSE_DEBUG` | 80 | Developer-detail |
| `MCA_BASE_VERBOSE_MAX` | 100 | Everything |

Set per-framework or per-component:

```
OMPI_MCA_coll_base_verbose=10           # coll selection events
OMPI_MCA_pml_base_verbose=100           # everything pml
OMPI_MCA_opal_common_ucx_verbose=10     # shared UCX init
```

### `opal_show_help`

[`opal/util/show_help.h:131`](../opal/util/show_help.h#L131). Prints formatted multi-line help from `.txt` files in `$prefix/share/openmpi/`. Each help file has named topics; example call:

```c
opal_show_help("help-mpi-runtime.txt", "mpi_init: invalid arg", true);
```

When debugging, grep `share/openmpi/help-*.txt` to find the message a user is seeing.

### Debugger interface (`ompi/debuggers/`)

OMPI exposes two structured DLL interfaces for parallel debuggers (TotalView, Arm DDT, gdb scripts):

- **`msgq_interface.h`** ([`ompi/debuggers/msgq_interface.h`](../ompi/debuggers/msgq_interface.h), 696 lines) — message queue introspection. Lets a debugger walk the list of posted receives, unmatched sends, and pending requests on each rank.
- **`mpihandles_interface.h`** ([`ompi/debuggers/mpihandles_interface.h`](../ompi/debuggers/mpihandles_interface.h), 882 lines) — handle resolution. Lets a debugger pretty-print `MPI_Comm`, `MPI_Request`, `MPI_Datatype`, `MPI_Win`, `MPI_File`.

Implementation: [`ompi/debuggers/ompi_debuggers.c`](../ompi/debuggers/ompi_debuggers.c) (193 lines). The debugger DLL is loaded by debuggers via `dlopen`; the canary symbol `MPIR_being_debugged` ([`ompi/debuggers/debuggers.h:52`](../ompi/debuggers/debuggers.h#L52)) signals "a debugger is attached". The MPIR-style attachment protocol uses `MPIR_Breakpoint` and `MPIR_proctable`.

### PERUSE

[`ompi/peruse/`](../ompi/peruse/) — an older but still-supported performance event API. Tools register callbacks for events like "request issued", "request completed". Not as widely used as MPI_T pvars, but present.

### `OMPI_MCA_<fw>_<comp>_show_load_errors`

For debugging "why isn't my component loading?", set the framework's `show_load_errors` MCA param. The framework will print why each DSO failed to dlopen — common causes: missing symbols, ABI mismatch, dlopen flag issues.

### Common starting points

| Symptom | Where to look |
|---------|---------------|
| Wrong PML selected | `OMPI_MCA_pml_base_verbose=10`. Check `pml/ucx` is reporting `OPAL_COMMON_UCX_SUPPORT_DEVICE`. |
| `pml/ucx` reports TRANSPORT not DEVICE | UCX found a HCA but no compatible transport for it. Check `UCX_TLS`, `ucx_info -d`. |
| Coll component not picked | `OMPI_MCA_coll_base_verbose=10` shows the priority each component reported. |
| `coll/ucc` not used | `OMPI_MCA_coll_ucc_priority=200`, confirm UCC was found at configure time (`ompi_info -c | grep ucc`). |
| Hang in `MPI_Finalize` | Likely a leaked request or a missing fence. Set `OMPI_MCA_pml_base_verbose=100` to see request lifecycle. |
| Hang in `MPI_Init` | PMIx fence — check launcher and `OMPI_MCA_pmix_base_verbose=100`. |
| GPU memory not detected | `OMPI_MCA_accelerator_base_verbose=100`; also `UCX_LOG_LEVEL=info` (UCX side). |
| Component fails to load | `OMPI_MCA_<fw>_base_show_load_errors=1`. |

---

## Top files for a debugger

The ~10 files anyone debugging OMPI on HPC-X workloads will end up opening.

| File | Why |
|------|-----|
| [`ompi/runtime/ompi_mpi_init.c`](../ompi/runtime/ompi_mpi_init.c) | `ompi_mpi_init` ([`:340`](../ompi/runtime/ompi_mpi_init.c#L340)) — entry point. Where every "fails to start" bug begins. |
| [`ompi/runtime/ompi_mpi_finalize.c`](../ompi/runtime/ompi_mpi_finalize.c) | `ompi_mpi_finalize` ([`:109`](../ompi/runtime/ompi_mpi_finalize.c#L109)) — every teardown bug. |
| [`ompi/instance/instance.c`](../ompi/instance/instance.c) | `ompi_mpi_instance_init` ([`:851`](../ompi/instance/instance.c#L851)) — opens every MCA framework. |
| [`ompi/communicator/comm_init.c`](../ompi/communicator/comm_init.c) | Communicator creation, ID allocation, coll selection. |
| [`ompi/mca/coll/base/coll_base_comm_select.c`](../ompi/mca/coll/base/coll_base_comm_select.c) | `mca_coll_base_comm_select` ([`:216`](../ompi/mca/coll/base/coll_base_comm_select.c#L216)) — per-comm vtable install. |
| [`ompi/mca/pml/ucx/pml_ucx.c`](../ompi/mca/pml/ucx/pml_ucx.c) + [`pml_ucx_component.c`](../ompi/mca/pml/ucx/pml_ucx_component.c) | UCX-backed PML; the hot path. |
| [`ompi/mca/coll/ucc/coll_ucc_module.c`](../ompi/mca/coll/ucc/coll_ucc_module.c) + [`coll_ucc_dtypes.h`](../ompi/mca/coll/ucc/coll_ucc_dtypes.h) | UCC bridge + datatype/op translation. |
| [`opal/mca/common/ucx/common_ucx.{h,c}`](../opal/mca/common/ucx/common_ucx.h) | Shared UCX init + verbose helpers + support_level. |
| [`opal/mca/base/mca_base_framework.{h,c}`](../opal/mca/base/mca_base_framework.h) | Framework register/open/close — when "my component isn't loaded", look here. |
| [`opal/runtime/opal_progress.c`](../opal/runtime/opal_progress.c) | `opal_progress` ([`:216`](../opal/runtime/opal_progress.c#L216)) — every async-progress bug ends here. |
| [`opal/util/output.c`](../opal/util/output.c) (and `.h`) | Verbose logging plumbing. |
| [`ompi/debuggers/ompi_debuggers.c`](../ompi/debuggers/ompi_debuggers.c) + [`debuggers.h`](../ompi/debuggers/debuggers.h) | MPIR / debugger interface. |

---

## Interactions with sibling libraries

OMPI participates in a three-way dependency with [UCX](https://github.com/openucx/ucx) (transport) and [UCC](https://github.com/openucx/ucc) (collectives). The same seam table appears in all three architecture docs.

### → UCX (https://github.com/openucx/ucx)

UCX is OMPI's primary fabric for p2p, RMA, and (transitively, via UCC) collectives.

- Components: `ompi/mca/pml/ucx`, `ompi/mca/osc/ucx`, `oshmem/mca/spml/ucx`, `opal/mca/common/ucx`.
- UCP entry points called: `ucp_init_version`, `ucp_worker_create`, `ucp_ep_create`, `ucp_tag_send_nbx`/`_nb`, `ucp_tag_recv_nbx`/`_nb`/`_nbr`, `ucp_put_nbx`, `ucp_get_nbx`, `ucp_atomic_op_nbx`, `ucp_worker_progress`, `ucp_request_check_status`, `ucp_ep_flush_nbx`, `ucp_mem_map`, `ucp_rkey_pack`/`_unpack`.
- Shared init via [`opal/mca/common/ucx/common_ucx.{h,c}`](../opal/mca/common/ucx/common_ucx.h) — one `ucp_context_h` shared across PML/OSC/SPML.
- Memory hooks: `opal/mca/memory/patcher` patches glibc; `opal_common_ucx.opal_mem_hooks` (config) controls whether OMPI's hooks or UCM's hooks own memory invalidation.

### → UCC (https://github.com/openucx/ucc)

UCC backs the `coll/ucc` component for collectives.

- Component: [`ompi/mca/coll/ucc/`](../ompi/mca/coll/ucc/).
- UCC entry points called: `ucc_init_version`, `ucc_context_create`, `ucc_team_create_post`/`_test`/`_destroy`, `ucc_collective_init`, `ucc_collective_post`, `ucc_collective_finalize`, `ucc_context_progress`.
- Datatype/op translation: [`coll_ucc_dtypes.h:19-114`](../ompi/mca/coll/ucc/coll_ucc_dtypes.h#L19-L114) — `ompi_datatype_2_ucc_dt[]`, `ompi_op_to_ucc_op_map[]`, `ompi_dtype_to_ucc_dtype()`.
- Fallback: when UCC returns `UCC_ERR_NOT_SUPPORTED` (e.g. for derived datatypes, non-predefined ops, asymmetric mem), `coll/ucc` returns `OMPI_ERR_NOT_SUPPORTED` and OMPI dispatches to the next coll component (typically `coll/tuned`).
- One UCC team per `ompi_communicator_t`. UCC team creation uses an OMPI-supplied OOB callback that drives PMIx-based allgather.

### → HCOLL (Mellanox vendor branches)

HCOLL is the older Mellanox collectives library. The corresponding `coll/hcoll` component lives on vendor branches (`mellanox/v5.0.x_hpcx` and friends), not on upstream `main`. Selection between HCOLL and UCC there is by priority — newer HPC-X tunes priorities to favour UCC.

### → PRRTE / PMIx

OMPI hands off process-launch and out-of-band signalling to PRRTE (the runtime environment) which itself drives PMIx (the protocol). Both are bundled in [`3rd-party/prrte/`](../3rd-party/prrte/) and [`3rd-party/openpmix/`](../3rd-party/openpmix/). At MPI_Init, OMPI's `opal/mca/pmix/external` connects to the PRRTE-launched daemon and runs the modex.

### Cross-library seam table (identical in all three docs)

| Direction | OMPI side | UCC side | UCX side |
|-----------|-----------|----------|----------|
| MPI p2p | [`ompi/mca/pml/ucx/pml_ucx.c`](../ompi/mca/pml/ucx/pml_ucx.c) | — | `src/ucp/tag/{eager_snd,tag_send}.c` |
| MPI RMA | [`ompi/mca/osc/ucx/`](../ompi/mca/osc/ucx/) | — | `src/ucp/rma/`, atomics |
| MPI collectives via UCC | [`ompi/mca/coll/ucc/`](../ompi/mca/coll/ucc/) | `src/core/ucc_coll.c`, `src/components/tl/ucp/` | `src/ucp/{tag,am,rma}/` |
| Shared UCX init for OMPI | [`opal/mca/common/ucx/`](../opal/mca/common/ucx/) | — | `src/ucp/api/ucp.h` |
| OSHMEM | [`oshmem/mca/spml/ucx/`](../oshmem/mca/spml/ucx/) | — | `src/ucp/`, `src/uct/` |
| OMPI datatype → UCC | [`ompi/mca/coll/ucc/coll_ucc_dtypes.h:19-114`](../ompi/mca/coll/ucc/coll_ucc_dtypes.h#L19-L114) | `src/ucc/api/ucc.h` (`ucc_datatype_t`) | — |
| UCC → UCX (tl/ucp) | — | `src/components/tl/ucp/tl_ucp_*` | `src/ucp/api/ucp.h` |

---

## Glossary

| Term | Expansion |
|------|-----------|
| BML | Byte Management Layer — multiplexes BTLs for `pml/ob1`. |
| BTL | Byte Transfer Layer — point-to-point HW transport plugin. |
| CID | Communicator ID — globally unique per `ompi_communicator_t`. |
| coll | Collective MCA framework. |
| DPM | Dynamic Process Management (MPI_Comm_spawn et al.). |
| HCOLL | Mellanox hierarchical collectives (legacy; UCC is the successor). |
| MCA | Modular Component Architecture — OMPI's plugin system. |
| modex | Module exchange — PMIx-mediated peer info publish/lookup at init. |
| MPI_T | MPI tools API: cvars + pvars. |
| mpirc / mca-params.conf | OMPI's user/system configuration files. |
| MTL | Matching Transport Layer — HW-matching alternative to BTL. |
| OMPI | The MPI layer of Open MPI. |
| OPAL | Open Portable Access Layer — portability foundation. |
| OSC | One-Sided Communication MCA framework. |
| OSHMEM | Open MPI's OpenSHMEM implementation. |
| PERUSE | Performance USe Engine — older instrumentation API. |
| PML | Point-to-point Messaging Layer. |
| PMIx | Process Management Interface (Exascale). |
| PRRTE | PMIx Reference RunTime Environment — OMPI's launcher. |
| pvar | MPI_T performance variable. |
| RMA | Remote Memory Access (MPI_Put/Get/Accumulate via MPI_Win). |
| sm | Shared memory. |
| smsc | Shared Memory Single-Copy (CMA / xpmem / knem). |
| SPML | Symmetric PML (OSHMEM analogue of PML). |
| UCC | Unified Collective Communications. |
| UCP | UCX's protocol layer. |
| UCT | UCX's transport layer. |
| UCX | Unified Communication X. |
| vprotocol | Virtual protocol wrapper for FT (`vprotocol/pessimist`). |
