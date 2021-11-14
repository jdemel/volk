# VOLK profile

VOLK loads kernels dynamically. The fastest kernel for a specific host depends on the CPU.
Obviously, a x86 CPU will only execute SSE, AVX, etc. kernels while ARM CPUs only execute NEON kernels.

VOLK depends on the contents of `.volk/volk_config` to load the fastest host dependent kernel at start-up.


This is a new approach to kernel profiling with more modern tools.

## profile dependencies

We use [benchmark](https://github.com/google/benchmark) to facilitate our profiling needs.
It is available on Ubuntu in the `libbenchmark-dev` package.


