from src.sdav.network.SDAV import SDAV
import logging

model = SDAV(verbosity=logging.INFO)
dataset = model.get_dataset('datasets/outdoor_kennedylong/*.ppm')
model.fit_dataset(dataset)
