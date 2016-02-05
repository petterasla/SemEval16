from glob import glob
import pandas as pd
from glove_transformer import GloveVectorizer


glove_fnames = glob('*.pkl')
glove_vecs = pd.read_pickle(glove_fnames[1])

print glove_vecs
print GloveVectorizer(glove_vecs)