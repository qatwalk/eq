from line_profiler import LineProfiler

from rbergomi_example import run_model  # noqa: F401
from rbergomi import rBergomiMCState

if __name__ == "__main__":
    # Create a LineProfiler object, specifying the methods to be profiled by line
    lprofiler = LineProfiler(rBergomiMCState.advance)
    # Wrap the entry function with the LineProfiler object, then run it.
    lp_wrapper = lprofiler(run_model)
    lp_wrapper()
    # Print the profiling results in milliseconds
    lprofiler.print_stats(output_unit=1e-03)
