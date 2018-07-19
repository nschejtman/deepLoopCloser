from model.Refactored import DA

n = 30
s = 41

model = DA([n, s ** 2], 2500)
model.fit_dataset('/Users/nicolas/projects/deepLoopCloser/Dataset/outdoor_kennedylong/*.ppm', verbose=True)

