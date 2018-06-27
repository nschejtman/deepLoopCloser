def overwrite_dict(write: dict, overwrite: dict, skip_nones: bool = False):
    new = write.copy()
    for k in overwrite.keys():
        if skip_nones and overwrite[k] is None:
            pass
        else:
            new[k] = overwrite[k]
    return new
