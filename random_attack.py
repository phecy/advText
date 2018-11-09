from preprocessors import DATASET_TO_PREPROCESSOR
import dictionaries
import argparse
from dataloaders import TextDataset, TextDataLoader
import torch
from models.WordCNN import WordCNN
from dataloaders import pad_text
from torch.autograd import Variable
from random import choice
import numpy as np
from random import sample
from random import shuffle
import copy

parser = argparse.ArgumentParser(description="Text classification attack")
parser.add_argument('--preprocess_level', type=str, default='word', choices=['word', 'char'])
parser.add_argument('--dictionary', type=str, default='WordDictionary', choices=['WordDictionary', 'AllCharDictionary'])
parser.add_argument('--max_vocab_size', type=int, default=50000)
parser.add_argument('--min_count', type=int, default=None)
parser.add_argument('--start_end_tokens', type=bool, default=False)
group = parser.add_mutually_exclusive_group()
group.add_argument('--vector_size', type=int, default=128, help='Only for rand mode')
group.add_argument('--wordvec_mode', type=str, default='glove', choices=['word2vec', 'glove'])
parser.add_argument('--min_length', type=int, default=5)
parser.add_argument('--max_length', type=int, default=300)
parser.add_argument('--sort_dataset', action='store_true')
parser.add_argument('--mode', type=str, default='non-static', choices=['rand', 'static', 'non-static', 'multichannel'])
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3,4,5])
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--attack_mode', type=str, default='random_attack', choices=['random_attack', 'ergodic_attack'])
parser.add_argument('--sample_num', type=int, default=1000000)

args = parser.parse_args()

Preprocessor = DATASET_TO_PREPROCESSOR["ag_news"]
preprocessor = Preprocessor("ag_news")
train_data, _, test_data = preprocessor.preprocess(level="word")

Dictionary = getattr(dictionaries, 'WordDictionary')
dictionary = Dictionary(args)
dictionary.build_dictionary(train_data)
test_dataset = TextDataset(test_data, dictionary, args.sort_dataset, args.min_length, args.max_length)
test_dataloader = TextDataLoader(dataset=test_dataset, dictionary=dictionary, batch_size=64)


model = WordCNN(n_classes=preprocessor.n_classes, dictionary=dictionary, args=args)
model = model.cuda()
model.load_state_dict(torch.load("./checkpoints/best_model.pth"))

model.eval()
PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)
test_texts = [([dictionary.indexer(token) for token in text], label)
              for text, label in test_data]

if args.min_length or args.max_length:
    test_texts = [(pad_text(text, PAD_IDX, args.min_length, args.max_length), label)
                  for text, label in test_texts]

text_lengths = [len(text) for text, label in test_texts]
longest_length = max(text_lengths)

processed_data = [(pad_text(text, pad=PAD_IDX, min_length=longest_length),int(label))
                  for text, label in test_texts]
num = len(processed_data)

if args.attack_mode == 'random_attack':

    for attack_round in range(100000):
        if attack_round == 0:
            count = 0
            index = []
            for text, label in processed_data:
                texts_tensor = Variable(torch.LongTensor([text]).cuda())
                outputs = model(texts_tensor)
                _, argmax = outputs.max(dim=1)
                correct = argmax.cpu().data[0] == label
                count += correct
                index.append(correct)
            print("Before attack, model acc:", count/num)

            processed_data = [processed_data[idx] for idx, i in enumerate(index) if i == True]
            print(len(processed_data))
        else:
            count = 0
            index = []
            attacked_samples = []
            for text, label in processed_data:
                attack_word = choice(range(50000))+2
                pos = choice(range(len(np.nonzero(np.array(text))[0])))
                input = copy.deepcopy(text)
                input.insert(pos, attack_word)
                texts_tensor = Variable(torch.LongTensor([input[:longest_length]]).cuda())
                outputs = model(texts_tensor)
                _, argmax = outputs.max(dim=1)
                result = argmax.cpu().data[0]
                correct = result == label
                if correct == False:
                    print_line = ''
                    for idx, word in enumerate(input[:longest_length]):
                        if dictionary.idx2word[word] != '<PAD>':
                            if idx == pos and attack_word == word:
                                idf = '$'+dictionary.idx2word[word]+'$'
                                print_line += idf
                                print_line += ' '
                            else:
                                print_line += dictionary.idx2word[word]
                                print_line += ' '
                    attacked_samples.append((print_line,label,result))
                count += correct
                index.append(correct)
            print("After attack_round: %d, model acc: %f" %(attack_round, count / num))
            if count / num == 0:
                break
            if len([i for i in index if i == False]) != 0:
                attacked_data = [processed_data[idx] for idx, i in enumerate(index) if i == False]
                avg_attacked_length = np.mean([len(np.nonzero(np.array(data))[0]) for data,label in attacked_data])
                print("Average length of attacked_samples: ", avg_attacked_length)
                if len(attacked_data) > args.sample_num:
                    vis = sample(attacked_samples, args.sample_num)
                    for o1, o2, o3 in vis:
                        print(o1)
                        print('True label: ', o2)
                        print('Disturbed result : ', o3)
                else:
                    for o1, o2, o3 in attacked_samples:
                        print(o1)
                        print('True label: ', o2)
                        print('Disturbed result : ', o3)
            else:
                print("No samples attacked!!!")
            print('/*********************************************************************/')

            processed_data = [processed_data[idx] for idx, i in enumerate(index) if i == True]

else:
    count = 0
    index = []
    for text, label in processed_data:
        texts_tensor = Variable(torch.LongTensor([text]).cuda())
        outputs = model(texts_tensor)
        _, argmax = outputs.max(dim=1)
        correct = argmax.cpu().data[0] == label
        count += correct
        index.append(correct)
    print("Before attack, model acc:", count / num)
    num = count

    processed_data = [processed_data[idx] for idx, i in enumerate(index) if i == True]
    print(len(processed_data))
    attack_words = list(range(50000))
    shuffle(attack_words)
    for attack_word in attack_words:
        count = 0
        for text, label in processed_data:
            attack_word +=  2
            input = copy.deepcopy(text)
            input.insert(1, attack_word)
            texts_tensor = Variable(torch.LongTensor([input[:longest_length]]).cuda())
            outputs = model(texts_tensor)
            _, argmax = outputs.max(dim=1)
            correct = argmax.cpu().data[0] != label
            count += correct
        print("Word: %s, attack success rate : %f" % (dictionary.idx2word[attack_word], count / num))
