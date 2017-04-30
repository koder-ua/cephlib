def raise_me(exc, tb):
    raise exc.with_traceback(tb)


def tostr(vl):
    if isinstance(vl, bytes):
        return vl.decode('utf8')
    return vl
