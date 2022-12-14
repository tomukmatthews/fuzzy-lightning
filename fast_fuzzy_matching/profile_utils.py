"""Latency profiling utils."""
import functools

import pyinstrument


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


import edit_distance as ed
from timeit import timeit

str1, str2 = ('beersteinbeer', 'stein')
print(timeit('ed.damerau_levenshtein(str1, str2)', number=100000, globals=globals()))