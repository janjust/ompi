InifiniBand / RoCE support
==========================

.. error:: TODO This section needs to be converted from FAQ Q&A style
           to regular documentation style.

How are InfiniBand / RoCE devices supported in Open MPI?
--------------------------------------------------------

Open MPI's support for InfiniBand and RoCE devices has changed over
time.

In the Open MPI |ompi_series| series, InfiniBand and RoCE devices are
supported via the UCX (``ucx``) PML.

.. note:: Prior versions of Open MPI also included the ``openib`` BTL
          for InfiniBand and RoCE devices.  Open MPI |ompi_series| no
          longer includes the ``openib`` BTL.

/////////////////////////////////////////////////////////////////////////

What is UCX?
------------

`UCX <https://openucx.org/>`_ is an open-source optimized
communication library which supports multiple networks, including
RoCE, InfiniBand, uGNI, TCP, shared memory, and others. UCX
mixes-and-matches transports and protocols which are available on the
system to provide optimal performance. It also has built-in support
for GPU transports (with CUDA and ROCm providers) which lets
RDMA-capable transports access the GPU memory directly.

/////////////////////////////////////////////////////////////////////////

How do I use UCX with Open MPI?
-------------------------------

If Open MPI includes UCX support, then UCX is enabled and selected by
default for InfiniBand and RoCE network devices; typically, no
additional parameters are required.  In this case, the network port
with the highest bandwidth on the system will be used for inter-node
communication, and shared memory will be used for intra-node
communication.  To select a specific network device to use (for
example, ``mlx5_0`` device port 1):

.. code-block::

   shell$ mpirun -x UCX_NET_DEVICES=mlx5_0:1 ...

It's also possible to force using UCX for MPI point-to-point and
one-sided operations:

.. code-block::

   shell$ mpirun --mca pml ucx --mca osc ucx ...

For OpenSHMEM, in addition to the above, it's possible to force using
UCX for remote memory access and atomic memory operations:

.. code-block::

   shell$ mpirun --mca pml ucx --mca osc ucx --mca scoll ucx --mca atomic ucx ...

/////////////////////////////////////////////////////////////////////////

What is RDMA over Converged Ethernet (RoCE)?
--------------------------------------------

RoCE (which stands for *RDMA over Converged Ethernet*) provides
InfiniBand native RDMA transport on top of lossless Ethernet data
links.

Since we're talking about Ethernet, there's no Subnet Manager, no
Subnet Administrator, no InfiniBand SL, nor any other InfiniBand
Subnet Administration parameters.

Connection management in RoCE is based on the OFED RDMACM (RDMA
Connection Manager) service:

* The OS IP stack is used to resolve remote (IP,hostname) tuples to
  a DMAC.
* The outgoing Ethernet interface and VLAN are determined according
  to this resolution.
* The appropriate RoCE device is selected accordingly.
* Network parameters (such as MTU, SL, timeout) are set locally by
  the RDMACM in accordance with kernel policy.

/////////////////////////////////////////////////////////////////////////

How do I know what MCA parameters are available for tuning MPI performance?
---------------------------------------------------------------------------

The ``ompi_info`` command can display all the parameters available for
any Open MPI component.  For example:

.. code-block::

   shell$ ompi_info --param pml ucx --level 9

.. important:: Unlike most other Open MPI components, the UCX PML
               mainly uses environment variables for run-time tuning
               |mdash| not Open MPI MCA parameters.  Consult `the UCX
               documentation
               <https://openucx.org/documentation/>`_ for details
               about what environment variables are available.

/////////////////////////////////////////////////////////////////////////

How do I tell Open MPI which IB Service Level to use?
-----------------------------------------------------

In order to tell the UCX PML which SL to use, the IB SL must be
specified using the ``UCX_IB_SL`` environment variable.  For example:

.. code-block::

   shell$ mpirun --mca pml ucx -x UCX_IB_SL=N ...

The value of IB SL ``N`` should be between 0 and 15, where 0 is the
default value.

/////////////////////////////////////////////////////////////////////////

How do I run Open MPI over RoCE?
--------------------------------

In order to use RoCE with the UCX PML, the relevant Ethernet port must
be specified using the ``UCX_NET_DEVICES`` environment variable.  For
example:

.. code-block::

   shell$ mpirun --mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 ...

UCX selects IPv4 RoCEv2 by default. If different behavior is needed,
you can set a specific GID index:

.. code-block::

   shell$ mpirun --mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_IB_GID_INDEX=1 ...

/////////////////////////////////////////////////////////////////////////

.. _faq-ib-locked-memory-label:

How do I set the locked memory limits?
--------------------------------------

InfiniBand and RoCE use RDMA, which requires communication buffers to
be locked ("pinned") in physical memory.  The amount of memory a
process may lock is limited by the ``memlock`` resource limit.  If
this limit is too low, memory registration can fail and MPI jobs may
abort or run slowly.  For most HPC installations, the limit should be
``unlimited``.

The limit is typically set in a file in ``/etc/security/limits.d/``
(or in ``/etc/security/limits.conf`` on older systems), for example:

.. code-block::

   * soft memlock unlimited
   * hard memlock unlimited

These files are applied by ``pam_limits`` when a user logs in.  MPI
processes launched by a resource manager (such as Slurm, PBS/Torque,
or LSF) are usually not started through a login session; they inherit
the limits of the resource manager's daemon on each node.  Make sure
those daemons are started with an unlimited locked memory limit (for
example, with ``LimitMEMLOCK=infinity`` in the daemon's systemd
service file), and restart them after changing it.  Some resource
managers also propagate the limits of the shell that submitted the
job (for example, Slurm's ``PropagateResourceLimits`` setting), so
the limit on the submitting node can matter as well.

To see the limit that MPI processes actually get, run ``ulimit -l``
through the same launch path as your application, for example:

.. code-block::

   shell$ mpirun -n 2 sh -c 'ulimit -l'

/////////////////////////////////////////////////////////////////////////

.. _faq-ib-troubleshoot-label:

I'm experiencing a problem with Open MPI on my InfiniBand / RoCE network; how do I troubleshoot and get help?
-------------------------------------------------------------------------------------------------------------

In order for us to help you, it is *most* helpful if you can run a few
steps before sending an e-mail to both perform some basic
troubleshooting and provide us with enough information about your
environment to help you.  Please include answers to the following
questions in your e-mail:

#. Which UCX and OpenFabrics version are you running?  Please specify
   where you got the software from (e.g., from the OpenFabrics and/or
   UCX community web sites, already included in your Linux
   distribution, downloaded from NVIDIA's web site, etc.).

#. What distro and version of Linux are you running?  What is your
   kernel version?

#. What is the output of the ``ibv_devinfo`` command on a known "good"
   node and a known "bad" node?

   .. note:: There must be at least one port listed as "PORT_ACTIVE"
             for Open MPI to work.  If there is not at least one
             PORT_ACTIVE port, something is wrong with your InfiniBand
             / RoCE environment and Open MPI will not be able to run.

#. What is the output of the ``ifconfig`` command on a known "good"
   node and a known "bad" node?

   .. note:: Note that some Linux distributions do not put
             ``ifconfig`` in the default path for normal users; look
             for it at ``/sbin/ifconfig`` or ``/usr/sbin/ifconfig``.

#. If running under Bourne shells, what is the output of the ``ulimit
   -l`` command?

   If running under C shells, what is the output of the ``limit | grep
   memorylocked`` command?

   .. note:: If the value is not ``unlimited``, see
             :ref:`faq-ib-locked-memory-label`.
