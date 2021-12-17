import os
import warnings
import torch
import torchtext.legacy as tt
from utils.load_data import shell, load_datasets, load_vi2en, load_words2num, load_ge2en

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Turn off annoying torchtext warnings about pending deprecations
warnings.filterwarnings("ignore", module="torchtext", category=UserWarning)

# Make splits for data
def split_datasets(fields, dir, val_split):
    def process_mt_data(dir):
        if 'tmp.src' not in os.listdir(dir): # need to do this on first run
            for file in [f'{dir}/train.src', f'{dir}/train.tgt']:
                # Make new filenames 
                val_file = file.replace('train', 'dev')
                tmp_file = file.replace('train', 'tmp')

                # Pop open all the files to work with
                file = open(file, 'r')
                val_file = open(val_file, 'w')  
                tmp_file = open(tmp_file, 'w')
                lines = file.readlines()
                val_sz = int(len(lines) * val_split)

                # Write to the validation set
                for i in range(val_sz):
                    val_file.write(lines[i])

                # Build new train set from scratch
                for j in range(val_sz, len(lines)):
                    tmp_file.write(lines[j])

                file.close()
                val_file.close()
                tmp_file.close()

            # Move the tmp contents to train
            shell(f"cp {dir}/tmp.src {dir}/train.src")
            shell(f"cp {dir}/tmp.tgt {dir}/train.tgt")        
    dir = 'data/' + dir
    if dir == 'data/vi2en': # handle the machine translation task            
        load_vi2en()
        process_mt_data(dir)
        # shell(f"rm -rf {dir}/tmp.*")
        print("Finished processing the Vietnamese to English specific part.")
    elif dir == 'data/ge2en': # handle the machine translation task            
        load_ge2en()
        process_mt_data(dir)
        # shell(f"rm -rf {dir}/tmp.*")
        print("Finished processing the German to English specific part.")        
    else:
        load_words2num()

    train_data, val_data, test_data = tt.datasets.TranslationDataset.splits(
        ('.src', '.tgt'), fields, path=f'./{dir}',
        train='train', validation='dev', test='test')
    return train_data, val_data, test_data

def process_data(dir, val_split=0.1, verbose=1):
    SRC = tt.data.Field(include_lengths=True,         # include lengths
                        batch_first=False,            # batches will be max_len x bsz
                        tokenize=lambda x: x.split(), # use split to tokenize
                    ) 
    TGT = tt.data.Field(include_lengths=False,
                        batch_first=False,            # batches will be max_len x bsz
                        tokenize=lambda x: x.split(), # use split to tokenize
                        init_token="<bos>",           # prepend <bos>
                        eos_token="<eos>")            # append <eos>
    fields = [('src', SRC), ('tgt', TGT)]

    train_data, val_data, test_data = split_datasets(fields, dir, val_split)

    # Build vocabulary
    SRC.build_vocab(train_data.src)
    TGT.build_vocab(train_data.tgt)

    if verbose:
        print (f"Size of src vocab: {len(SRC.vocab)}")
        print (f"Size of tgt vocab: {len(TGT.vocab)}")
        print (f"Index for src padding: {SRC.vocab.stoi[SRC.pad_token]}")
        print (f"Index for tgt padding: {TGT.vocab.stoi[TGT.pad_token]}")
        print (f"Index for start of sequence token: {TGT.vocab.stoi[TGT.init_token]}")
        print (f"Index for end of sequence token: {TGT.vocab.stoi[TGT.eos_token]}")

    ## Batch the data

    BATCH_SIZE = 32     # batch size for training and validation
    TEST_BATCH_SIZE = 1 # batch size for test; we use 1 to make implementation easier

    train_iter, val_iter = tt.data.BucketIterator.splits((train_data, val_data),
                                                        batch_size=BATCH_SIZE, 
                                                        device=device,
                                                        repeat=False, 
                                                        sort_key=lambda x: len(x.src), # sort by length to minimize padding
                                                        sort_within_batch=True)
    test_iter = tt.data.BucketIterator(test_data, 
                                    batch_size=TEST_BATCH_SIZE, 
                                    device=device,
                                    repeat=False, 
                                    sort=False, 
                                    train=False)

    return SRC, TGT, train_iter, val_iter, test_iter
