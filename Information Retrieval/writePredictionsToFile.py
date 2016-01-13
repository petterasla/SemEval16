# Tools for writing predictions to file
def initFile(filename):
    file = open(filename+".txt", 'w')
    return file

def writePrdictionToFile(ID, TARGET, TWEET, STANCE, file):
    s = ID + "\t" + TARGET + "\t" + TWEET + "\t" + STANCE + "\n"
    s = s.encode("utf-8")
    file.write(s)

# Example of usage
"""
file = initFile("predictions")
writePrdictionToFile("1","Hillary Clinton","bfivblbvirbbdbddnnf fej cr reif eri crierknef","FAVOR", file)
writePrdictionToFile("2","Hillary Clinton","bfivblbvirbbdbddnnf fej cr reif eri crierknef","AGAINST", file)
writePrdictionToFile("3","Hillary Clinton","bfivblbvirbbdbddnnf fej cr reif eri crierknef","NONE", file)
file.close()
"""