from utils.DictMerger import overwrite_dict


def test_merge_nothing():
    dict1 = {'a': 0}
    dict2 = {}
    result = overwrite_dict(dict1, dict2)
    assert result == dict1


def test_merge_something():
    dict1 = {'a': 0}
    dict2 = {'b': 1}
    result = overwrite_dict(dict1, dict2)
    assert result['a'] == 0
    assert result['b'] == 1


def test_merge_nones():
    dict1 = {'a': 0}
    dict2 = {'a': None}
    result = overwrite_dict(dict1, dict2)
    assert result['a'] is None


def test_skip_nones():
    dict1 = {'a': 0}
    dict2 = {'a': None}
    result = overwrite_dict(dict1, dict2, skip_nones=True)
    assert result['a'] == 0
