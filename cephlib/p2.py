def raise_me(exc, tb):
    raise exc, None, tb


def tostr(vl):
    if isinstance(vl, unicode):
        return vl.encode('utf8')
    return vl
