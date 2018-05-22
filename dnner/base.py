import builtins


try:
    profile = builtins.__dict__['profile']
except KeyError:
    # No line profiler, provide a pass-through version
    def profile(func): return func

class Prior():
    pass

class Activation():
    pass

class Ensemble():
    pass
