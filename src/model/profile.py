"""
Profile the rBergomi model to find timing bottlenecks in the code.
"""

from line_profiler import LineProfiler
from rbergomi import rBergomiMC
from rbergomi_example import run_model  # noqa: F401

if __name__ == "__main__":
    # Create a LineProfiler object, specifying the methods to be profiled by line
    lprofiler = LineProfiler(rBergomiMC.advance)
    # Wrap the entry function with the LineProfiler object, then run it.
    lp_wrapper = lprofiler(run_model)
    lp_wrapper()
    # Print the profiling results in milliseconds
    lprofiler.print_stats(output_unit=1e-03)
