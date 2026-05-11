# Open MPI — Claude Code entry point

Concise orientation for AI agents and engineers landing in this repo. Deep architectural reference lives at [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Read that next.

## What this is

Open MPI = a high-performance, vendor-neutral implementation of MPI 4.0+ and OpenSHMEM, organised around the Modular Component Architecture (MCA) — almost every algorithm, transport, and policy is a runtime-pluggable component. Ships with bundled PMIx, PRRTE, hwloc, and libevent.

The codebase is layered: **OPAL** (portability) → **OMPI** (MPI) → **OSHMEM** (SHMEM). Each layer has its own `mca/` subdirectory.

## Stack at a glance (one screen)

```
   user MPI program
          │
   ┌──────▼─────────────────────────────────────────────┐
   │ ompi/  — MPI layer                                 │
   │  ompi/runtime/ompi_mpi_init.c  ← entry             │
   │  ompi/communicator/comm.c      ← MPI_Comm          │
   │  ompi/mca/{pml,coll,osc,mtl,io,fcoll,fbtl,fs,…}/   │
   │       pml/ucx, coll/ucc, osc/ucx ← HPC-X path      │
   │       pml/ob1, coll/tuned, osc/rdma ← SW fallback  │
   └──────┬─────────────────────────────────────────────┘
          │
   ┌──────▼─────────────────────────────────────────────┐
   │ opal/  — portability                               │
   │  opal/mca/{btl,mpool,rcache,pmix,threads,…}/       │
   │  opal/mca/common/ucx/  ← shared UCX init           │
   │  opal/mca/accelerator/{cuda,rocm,ze}/  ← GPU       │
   │  opal/runtime/opal_progress.c ← async engine       │
   └──────┬─────────────────────────────────────────────┘
          │
   ┌──────▼─────────────────────────────────────────────┐
   │ 3rd-party/  bundled                                │
   │   openpmix/, prrte/, hwloc, libevent               │
   └────────────────────────────────────────────────────┘

   oshmem/  ← OpenSHMEM layer (uses spml/ucx, scoll/ucc)
```

## Where to look first

| Symptom | Start here |
|---------|-----------|
| `MPI_Init` hang | [`ompi/runtime/ompi_mpi_init.c:340`](ompi/runtime/ompi_mpi_init.c#L340) → [`ompi/instance/instance.c:851`](ompi/instance/instance.c#L851). Likely PMIx fence; set `OMPI_MCA_pmix_base_verbose=100`. |
| `MPI_Finalize` hang | [`ompi/runtime/ompi_mpi_finalize.c:109`](ompi/runtime/ompi_mpi_finalize.c#L109). Leaked request or premature framework close. |
| Wrong PML chosen | `OMPI_MCA_pml_base_verbose=10`. Then [`ompi/mca/pml/base/`](ompi/mca/pml/base/) and the component's `*_component.c`. |
| `pml/ucx` not selected | Check `opal_common_ucx_support_level()` in [`opal/mca/common/ucx/common_ucx.h:103-113`](opal/mca/common/ucx/common_ucx.h#L103-L113) — needs `OPAL_COMMON_UCX_SUPPORT_DEVICE`. |
| Wrong coll component | `OMPI_MCA_coll_base_verbose=10` then [`ompi/mca/coll/base/coll_base_comm_select.c:216`](ompi/mca/coll/base/coll_base_comm_select.c#L216). |
| `coll/ucc` falling back to `coll/tuned` | UCC returned `UCC_ERR_NOT_SUPPORTED`. Check datatype/op via [`coll_ucc_dtypes.h`](ompi/mca/coll/ucc/coll_ucc_dtypes.h). |
| Component not loading | Set `OMPI_MCA_<fw>_base_show_load_errors=1`. Then [`opal/mca/base/mca_base_component_find.c`](opal/mca/base/mca_base_component_find.c). |
| GPU memory not detected | [`opal/mca/accelerator/`](opal/mca/accelerator/), `OMPI_MCA_accelerator_base_verbose=100`. |
| Async progress missing | [`opal/runtime/opal_progress.c:216`](opal/runtime/opal_progress.c#L216). |
| MCA framework lifecycle | [`opal/mca/base/mca_base_framework.h:128-161`](opal/mca/base/mca_base_framework.h#L128-L161) and `MCA_BASE_FRAMEWORK_DECLARE` at [`:267`](opal/mca/base/mca_base_framework.h#L267). |
| Verbose / log output | [`opal/util/output.h`](opal/util/output.h); levels at [`opal/mca/base/base.h:81-100`](opal/mca/base/base.h#L81-L100). |

## Key entry points

| Path | Why |
|------|-----|
| [`ompi/runtime/ompi_mpi_init.c:340`](ompi/runtime/ompi_mpi_init.c#L340) | `ompi_mpi_init` |
| [`ompi/runtime/ompi_mpi_finalize.c:109`](ompi/runtime/ompi_mpi_finalize.c#L109) | `ompi_mpi_finalize` |
| [`ompi/instance/instance.c:851`](ompi/instance/instance.c#L851) | `ompi_mpi_instance_init` — opens MCA frameworks |
| [`ompi/communicator/comm_init.c`](ompi/communicator/comm_init.c) + [`comm.c`](ompi/communicator/comm.c) | Communicator creation, ID alloc |
| [`ompi/mca/coll/base/coll_base_comm_select.c:216`](ompi/mca/coll/base/coll_base_comm_select.c#L216) | Per-`MPI_Comm` coll vtable install |
| [`ompi/mca/pml/ucx/pml_ucx.c`](ompi/mca/pml/ucx/pml_ucx.c):636/899/1022 | `mca_pml_ucx_irecv` / `_isend` / `_send` |
| [`opal/runtime/opal_progress.c:216`](opal/runtime/opal_progress.c#L216) | `opal_progress` — the async engine |
| [`opal/mca/common/ucx/common_ucx.h:88-96`](opal/mca/common/ucx/common_ucx.h#L88-L96) | `opal_common_ucx_module_t` — shared UCX state |

## Key env vars

| Var | Effect |
|-----|--------|
| `OMPI_MCA_pml=ucx` | Force PML selection. `^ob1` to exclude. |
| `OMPI_MCA_coll=ucc,tuned,basic` | Coll fallback chain. |
| `OMPI_MCA_<fw>_base_verbose=10` | Selection-event verbosity (level 10–100). |
| `OMPI_MCA_<fw>_base_show_load_errors=1` | Print why a DSO failed to load. |
| `OMPI_MCA_opal_common_ucx_verbose=10` | Shared UCX init log. |
| `OMPI_MCA_opal_common_ucx_tls=rc,sm,self` | Restrict UCX TLs. |
| `OMPI_MCA_opal_common_ucx_devices=mlx5_0:1` | Restrict UCX devices. |
| `OMPI_MCA_btl=^openib` | Exclude legacy BTLs (HPC-X uses `pml/ucx`, not BTLs). |
| `--mca <fw>_<comp>_<param> <val>` | Same as `OMPI_MCA_*` but on the mpirun command line. |
| `--display-map`, `--report-bindings` | PRRTE rank-placement / core-binding. |

## Build

OMPI uses **`autogen.pl`**, not `autogen.sh` — it's a 1819-line Perl script that discovers all frameworks/components and generates configure inputs.

```bash
./autogen.pl
./configure --prefix=/opt/openmpi \
            --with-ucx=/opt/ucx \
            --with-ucc=/opt/ucc \
            --with-hcoll=/opt/hcoll \
            --with-cuda=/usr/local/cuda \
            --with-pmix=internal --with-prrte=internal \
            --with-libevent=internal --with-hwloc=internal
make -j && make install
```

Repo also ships [`scripts/build-ompi.sh`](../scripts/build-ompi.sh) (in the parent `hpcx-stack-dev/` repo) for HPC-X-friendly builds.

If you add a framework or component, **rerun `autogen.pl`** before `configure`.

See [`docs/ARCHITECTURE.md` § Build system](docs/ARCHITECTURE.md#build-system) and [`HACKING.md`](HACKING.md).

## MCA framework lifecycle (the contract you must respect)

```
register   ─▶  open  ─▶  select  ─▶  close
   │           │           │           │
component params      query each       reverse order in finalize
registered with var   component for
registry              priority + module
```

```c
MCA_BASE_FRAMEWORK_DECLARE(opal, foo, "description",
                           opal_foo_register, opal_foo_open, opal_foo_close,
                           opal_foo_static_components, MCA_BASE_FRAMEWORK_FLAG_LAZY);
```

See [`opal/mca/base/mca_base_framework.h`](opal/mca/base/mca_base_framework.h).

## Conventions worth knowing before editing

- **OPAL is not MPI-aware.** No symbol from OPAL may include from `ompi/`. The dependency is one-way (OMPI → OPAL).
- **`mca_base_component_2_1_0_t` first in component descriptors.** Frameworks subclass; the base must be the first member or DSO loading breaks.
- **Verbosity envelope.** Don't print at level 0–10 unless it's an error or a one-time selection event. Trace and debug levels (60, 80) are for development.
- **`opal_show_help` for user-facing errors.** Don't `fprintf(stderr, …)` — write a `help-*.txt` file.
- **Reference counting via opal class system.** `OBJ_NEW(type)` / `OBJ_RELEASE(obj)` for allocation; `OBJ_RETAIN(obj)` to share. See [`opal/class/`](opal/class/).
- **Threading.** Wrap shared-state access in `OPAL_THREAD_LOCK(mtx)` / `OPAL_THREAD_UNLOCK(mtx)`. These are no-ops in single-threaded builds.
- **`autogen.pl` not `autogen.sh`.** Saying it twice because new contributors get this wrong every time.

## Sibling repos

OMPI is co-developed with:

- [UCX](https://github.com/openucx/ucx) — primary p2p/RMA fabric via `pml/ucx`, `osc/ucx`, `spml/ucx`.
- [UCC](https://github.com/openucx/ucc) — collective backend via `coll/ucc`.

See [`docs/ARCHITECTURE.md` § Interactions with sibling libraries](docs/ARCHITECTURE.md#interactions-with-sibling-libraries) for the full seam table.

## Don't break these without thinking hard

- Public MPI API in `ompi/include/mpi.h.in` and `ompi/mpi/c/`. Standardised; breaking changes need WG approval.
- MCA component ABI versions in `opal/mca/mca.h` and per-framework `<framework>.h`. Bumping forces a rebuild of every consumer of out-of-tree components.
- The MCA parameter system at [`opal/mca/base/mca_base_var.h`](opal/mca/base/mca_base_var.h) — operator runbooks worldwide reference these names.
- The debugger DLL interface in [`ompi/debuggers/{msgq,mpihandles}_interface.h`](ompi/debuggers/) — TotalView and DDT depend on it.
- `MPIR_Breakpoint` / `MPIR_being_debugged` symbols in [`ompi/debuggers/debuggers.h`](ompi/debuggers/debuggers.h) — every parallel debugger looks for these.
