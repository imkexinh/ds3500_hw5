from collections import defaultdict
import time


def profile(f):
    """ Convenience method to make decorator tags simpler
    e.g., @profile instead of @Profiler.profile """
    return Profiler.profile(f)


class Profiler:
    """ A code profiling class.  Keep track of function calls and runtime """
    calls = defaultdict(int)   # default = 0,  function name --> # of calls of that function
    time = defaultdict(float)  # default = 0.0 function name --> total elapsed time in that function

    @staticmethod
    def profile(f):
        """ The profiling decorator """

        def wrapper(*args, **kwargs):
            start = time.time_ns()
            val = f(*args, **kwargs)  # the function is called
            end = time.time_ns()
            elapsed = (end - start) / 10 ** 9

            Profiler.calls[f.__name__] += 1
            Profiler.time[f.__name__] += elapsed
            return val

        return wrapper


    @staticmethod
    def report():
        """ Summarize # calls, total runtime, and time/call for each function """
        print("Function              Calls     TotSec   Sec/Call")
        for name, num in Profiler.calls.items():
            sec = Profiler.time[name]
            print(f'{name:20s} {num:6d} {sec:10.6f} {sec / num:10.6f}')



