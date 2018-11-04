from pynn.tensor import Tensor

def size(*args: Tensor):
    if not len(args):
        raise ValueError
    
    n = args[0].size
    
    for t in args[1:]:
        if t.size != n:
            raise ValueError

    return n