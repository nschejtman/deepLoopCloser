import os

import numpy as np


class Writer:
    @staticmethod
    def run_save(parser, reader):
        n_batches = len(reader)
        for i, batch in enumerate(reader):
            print("Processing %d/%d" % (i + 1, n_batches))
            parsed_batch = np.array(parser.calculate__all_from_path(batch))
            save_dir = Writer._init_save_dirs(n_batches, parser, reader)
            save_file = "%s/[batch=%s].csv" % (save_dir, Writer._batch_file_number_string(n_batches, i))
            np.savetxt(save_file, parsed_batch, delimiter=",")

    @staticmethod
    def _init_save_dirs(n_batches, parser, reader):
        save_base = "%s_parsed" % reader.directory
        if not os.path.exists(save_base):
            os.mkdir(save_base)
        save_dir = "%s/[n=%d][p=%d]" % (save_base, parser.n_patches, parser.patch_size)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = "%s/[batches=%d]" % (save_dir, n_batches)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        return save_dir

    @staticmethod
    def _batch_file_number_string(n_batches, i):
        n_digits = len(str(n_batches))
        return str(i).zfill(n_digits)
