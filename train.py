from src.sdav.network.SDAV import SDAV

model = SDAV()
model.fit('datasets/outdoor_kennedylong/*.ppm')
