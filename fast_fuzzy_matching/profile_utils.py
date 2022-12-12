import functools
from timeit import timeit

import pyinstrument
import pylcs

import lcs

# print(timeit(lambda: lcs.longest_common_substring_length('tomseantom', 'msea'), number=1000000))
A = 'We are shannonai'
B = 'We like shannonai'

print('pylcs: ', timeit(lambda: pylcs.lcs_string_length(A, B), number=1000000))
print('mine: ', timeit(lambda: lcs.longest_common_substring_length(A, B), number=1000000))


def profile(func):
    """Decorator for profiling functions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler_kwargs = dict(
            pyinst_interval=0.0001, pyinst_color=True, pyinst_show_all=True, pyinst_timeline=False, cprof_sort='cumtime'
        )
        profiler = pyinstrument.Profiler(interval=profiler_kwargs['pyinst_interval'])

        profiler.start()
        retval = func(*args, **kwargs)
        profiler.stop()

        prof_output = profiler.output_text(
            color=profiler_kwargs['pyinst_color'],
            show_all=profiler_kwargs['pyinst_show_all'],
            timeline=profiler_kwargs['pyinst_timeline'],
        )
        print(prof_output)
        return retval

    return wrapper
