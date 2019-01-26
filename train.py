import logging

from src.sdav.network.SDAV import SDAV

model = SDAV(verbosity=logging.INFO)
dataset = model.get_dataset('datasets/outdoor_kennedylong/*.ppm')
model.fit_dataset(dataset)
