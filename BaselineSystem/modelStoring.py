import pickle

def storeModel(model, filename):
    """
    Stores the fitted model in an file

    :param model:    scikit learn model (like SVC)
    :param filename  String with name of file to store model in
    """
    with open(filename+'.pkl', 'wb') as fid:
        pickle.dump(model, fid)


def loadModel(filename):
    """
     Loads stored model from file

    :param filename:     Name of file where model is stored
    :return:             The scikit learn model
    """
    with open(filename+'.pkl', 'rb') as fid:
        model = pickle.load(fid)
    return model