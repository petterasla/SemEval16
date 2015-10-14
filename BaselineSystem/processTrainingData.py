def processData(doc_name):
    data = []
    f = open(doc_name,"r")
    line = f.readline()     #Firste line states id, year, title of the article
    line = f.readline()     #Starts reading from the second line
    while line:
        dataLine = []
        index = line.index("\t")                # Finding the index of the first tab \t
        id = line[0:index]                      # Storing the ID
        dataLine.append(id)                     # Adding the ID to a temporary list
        while index < len(line):                # Looping through the rest of the tabs: \t
            index2 = line.find("\t", index+1)
            if index == -1:
                dataLine[3] = dataLine[3][:-1]  # Removing the last \r on each stance
                break
            attr = line[index+1:index2]         # Storing each attr in the line (topic, tweet and stance)
            dataLine.append(attr)               # Adding the attr to the temp list
            index = index2                      # Setting prev index to newest

        data.append(dataLine)
        line = f.readline()
    f.close()
    return data

# data = processData("semeval2016-task6-trainingdata.txt")
    #
    # print "Lenght of the data: " +str(len(data))
    # print "Test sample from the topic: Atheism"
    # print data[1]
    # print
    # print "Test sample from the topic: Climate Change is a Real Concern"
    # print data[620]
    # print
    # print "Test sample from the topic: Feminist Movement"
    # print data[920]
    # print
    # print "Test sample from the topic: Hillary Clinton"
    # print data[1700]
    # print
    # print "Test sample from the topic: Legalization of Abortion"
    # print data[2400]