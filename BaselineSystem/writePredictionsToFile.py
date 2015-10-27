# Tools for writing predictions to file
def initFile(filename):
    file = open(filename+".txt", 'w')
    file.write("ID\tTARGET\tTWEET\tSTANCE\n")
    return file

def writePrdictionToFile(ID, TARGET, TWEET, STANCE, file):
    file.write(ID + "\t" + TARGET + "\t" + TWEET + "\t" + STANCE + "\n")

# Example of usage
"""
file = initFile("predictions")
writePrdictionToFile("1","Hillary Clinton","bfivblbvirbbdbddnnf fej cr reif eri crierknef","FAVOR", file)
writePrdictionToFile("2","Hillary Clinton","bfivblbvirbbdbddnnf fej cr reif eri crierknef","AGAINST", file)
writePrdictionToFile("3","Hillary Clinton","bfivblbvirbbdbddnnf fej cr reif eri crierknef","NONE", file)
file.close()
"""