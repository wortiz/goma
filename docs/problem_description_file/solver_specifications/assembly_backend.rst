Assembly Backend
================

::

   Assembly Backend = {legacy | kokkos}

Select the assembly implementation for the complete coupled system. The default
is ``legacy``, which preserves the existing assembly behavior and works with
``Use AutoDiff Assembly`` as before.

``kokkos`` reserves the experimental device assembly path. Build its
infrastructure with ``-DENABLE_KOKKOS_ASSEMBLY=ON``; this requires C++20 and a
Trilinos installation containing Kokkos, Tpetra, and Stratimikos, with the latter
two enabled in Goma. Legacy-only builds continue to use C++17 and C11.

The initial infrastructure does not yet implement flow assembly kernels.
Selecting ``kokkos`` therefore reports an error before assembly, including in
an enabled build. It never falls back to legacy assembly. Kokkos with
``Use AutoDiff Assembly = yes`` is also unavailable. Runtime ownership,
capability checks for individual physics, and numerical qualification will be
added with the assembly implementation.
