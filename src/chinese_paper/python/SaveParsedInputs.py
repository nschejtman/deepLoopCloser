from input.CvInputParser import Parser
from input.ParsedInputWriter import Writer
from utils.BufferedFileReader import BufferedReader

reader = BufferedReader("../../../Dataset/outdoor_kennedylong", ".ppm", 9)
parser = Parser(30, 40)
Writer.run_save(parser, reader)
