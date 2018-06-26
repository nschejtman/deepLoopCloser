import numpy as np
from pathlib2 import Path

from utils.BufferedFileReader import BufferedReader

root = Path('../..')


def test_reader_matching_batches():
    base_dir = '%s/resources/sample_dir' % root
    batch_size = 2
    reader = BufferedReader(base_dir, '.test', batch_size)
    assert len(reader) == 5
    i = 0
    for batch in reader:
        resolved_base_dir = str(Path(base_dir).resolve())
        expected = ['%s/file%d.test' % (resolved_base_dir, i * batch_size),
                    '%s/file%d.test' % (resolved_base_dir, i * batch_size + 1)]
        assert np.array_equal(batch, expected)
        i += 1


def test_reader_non_matching_batches():
    base_dir = '%s/resources/sample_dir' % root
    batch_size = 3
    reader = BufferedReader(base_dir, '.test', batch_size)
    assert len(reader) == 3
    i = 0
    for batch in reader:
        resolved_base_dir = str(Path(base_dir).resolve())
        expected = ['%s/file%d.test' % (resolved_base_dir, i * batch_size),
                    '%s/file%d.test' % (resolved_base_dir, i * batch_size + 1),
                    '%s/file%d.test' % (resolved_base_dir, i * batch_size + 2)]
        assert np.array_equal(batch, expected)
        i += 1
