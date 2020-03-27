#coding:utf-8
import jieba
import re

fmt_pat=re.compile('[^一-龥]+')

class CorpusLoarder():
    def __init__(self,path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            yield line

class VocabBuilder():
    def __init__(self,
                train_path = '../data/train.txt',
                dev_path = '../data/dev.txt',
                test_path = '../data/test.txt'):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.vocabs, self.word2id = self.main()

    def segger(self, text):
        '''
        seg word using jieba
        '''
        return list(jieba.cut(text))

    def formater(self, text):
        '''
        remove ugly words not Chinese
        '''
        return fmt_pat.sub('',text)

    def build_vocab(self, path):
        corpus_train=CorpusLoarder(path)
        vocabs = set() #使用集合来保存词典
        for line in corpus_train:
            line = line.strip().split('\t')
            text_a, text_b = line[0], line[1]
            text_a, text_b  = self.formater(text_a),self.formater(text_b)
            if text_a and text_b:
                text_a_words, text_b_words = set(self.segger(text_a)), set(self.segger(text_b))
                vocabs.update(text_a_words)
                vocabs.update(text_b_words)
        return vocabs

    def main(self):
        vocabs, word2id = set(), {}
        vocabs_train = self.build_vocab(self.train_path)
        vocabs_dev = self.build_vocab(self.dev_path)
        vocab_test = self.build_vocab(self.test_path)
        vocabs.update(vocabs_train)
        vocabs.update(vocab_test)
        vocabs.update(vocabs_dev)
        for id,word in enumerate(vocabs):
            word2id[word] = id
        print('vocab size: %d'%len(vocabs))
        return vocabs, word2id

class Transform(VocabBuilder):
    '''
    transform text to ids
    '''
    def __init__(self):
        super(Transform,self).__init__()

    def text2id(self, text):
        word_ids = []
        if text:
            fmt_text = self.formater(text)
            if fmt_text:
                seg_words = self.segger(fmt_text)
                if seg_words:
                    for word in seg_words:
                        id = self.word2id.get(word,None)
                        if id:
                            word_ids.append(str(id))
        return word_ids

    def batch_transform(self, from_path, to_path):
        with open(from_path) as fr:
            with open(to_path,'w') as fw:
                for line in fr.readlines():
                    line = line.strip().split('\t')
                    text_a, text_b, label = line[0],line[1],line[2]
                    text_a_ids = self.text2id(text_a)
                    text_b_ids = self.text2id(text_b)
                    if text_a_ids and text_b_ids:
                        fw.write(' '.join(text_a_ids)+'\t'+' '.join(text_b_ids)+'\t'+label+'\n')

if __name__=='__main__':
    transformer = Transform()
    transformer.batch_transform('../data/train.txt','../data/train_ids.txt')
    transformer.batch_transform('../data/dev.txt','../data/dev_ids.txt')
    transformer.batch_transform('../data/test.txt','../data/test_ids.txt')