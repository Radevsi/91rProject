# Please do not change this cell because some hidden tests might depend on it.
import os

DIRS = ['train.src', 'train.tgt', 'test.src', 'test.tgt']


# # Define custom function for shell commands
def shell(commands, warn=True):
    """Executes the string `commands` as a sequence of shell commands.
     
       Prints the result to stdout and returns the exit status. 
       Provides a printed warning on non-zero exit status unless `warn` 
       flag is unset.
    """
    file = os.popen(commands)
    print (file.read().rstrip('\n'))
    exit_status = file.close()
    if warn and exit_status != None:
        print(f"Completed with errors. Exit status: {exit_status}\n")
    return exit_status

# Download data

def get_path(dir_name):
  """
  Returns the entire path to the data directory. 
  Creates directory if it does not exist.
  """
  path = os.getcwd() + f'/data/{dir_name}'
  try:
    os.listdir(path)
  except FileNotFoundError:
    os.mkdir(path)
  return path


def load_words2num(words2num_dir="words2num"):
  """Load the dataset from lab4-4 in CS187"""

  print("Loading words2num dataset...")
  path = get_path(words2num_dir)
  if not set(DIRS).issubset(set(os.listdir(path))):
    shell(f"""
      wget -nv -N -P {path} https://raw.githubusercontent.com/nlp-course/data/master/Words2Num/train.src
      wget -nv -N -P {path} https://raw.githubusercontent.com/nlp-course/data/master/Words2Num/train.tgt
      wget -nv -N -P {path} https://raw.githubusercontent.com/nlp-course/data/master/Words2Num/dev.src
      wget -nv -N -P {path} https://raw.githubusercontent.com/nlp-course/data/master/Words2Num/dev.tgt
      wget -nv -N -P {path} https://raw.githubusercontent.com/nlp-course/data/master/Words2Num/test.src
      wget -nv -N -P {path} https://raw.githubusercontent.com/nlp-course/data/master/Words2Num/test.tgt
    """)
    print("Successfully downloaded Words2Num dataset!")
  else:
    print("Words2Num dataset already loaded!")

def load_vi2en(vi2en_dir="vi2en"):
  """Load the dataset used in homework 3 of 6864"""

  print("Loading vi2en dataset...")
  path = get_path(vi2en_dir)
  if not set(DIRS).issubset(set(os.listdir(path))):
    shell(f""" 
      wget -nv -O {path}/train.tgt https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en
      wget -nv -O {path}/train.src https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi
      wget -nv -O {path}/test.tgt https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en
      wget -nv -O {path}/test.src https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi
    """)    
    print("Successfully downloaded Vietnamese-to-English MT dataset!")
  else:
    print("Vi2En dataset already loaded!")

def load_ge2en(ge2en_dir="ge2en"):
  """Load the dataset used in homework 3 of 6864"""

  print("Loading ge2en dataset...")
  path = get_path(ge2en_dir)
  if not set(DIRS).issubset(set(os.listdir(path))):
    shell(f""" 
      wget -nv -O {path}/train.tgt https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
      wget -nv -O {path}/train.src https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
      wget -nv -O {path}/test.tgt https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en
      wget -nv -O {path}/test.src https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de
    """)
    print("Successfully downloaded German-to-English MT dataset!")
  else:
    print("Ge2En dataset already loaded!")


def load_datasets():
  print("Loading all datasets...")
  load_words2num()
  load_vi2en()
  load_ge2en()

if __name__ == '__main__':
  load_datasets()