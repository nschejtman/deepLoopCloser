import os
import numpy as np


class Writer:
    @staticmethod
    def run_save(parser, reader):
        n_batches = len(reader)
        for i, batch in enumerate(reader):
            print("Processing %d/%d" % (i + 1, n_batches))
            parsed_batch = parser.calculate__all_from_path(batch)
            reshaped = np.array(parsed_batch).reshape(len(batch) * parser.n_patches, parser.patch_size ** 2)
            save_base = "%s_parsed" % reader.directory
            if not os.path.exists(save_base):
                os.mkdir(save_base)
            save_dir = "%s/[n=%d][p=%d]" % (save_base, parser.n_patches, parser.patch_size)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_dir = "%s/[batches=%d]" % (save_dir, n_batches)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            n_digits = len(str(n_batches))
            np.savetxt("%s/[batch=%s].csv" % (save_dir, str(i).zfill(n_digits)), reshaped, delimiter=",")
