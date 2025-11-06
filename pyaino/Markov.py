import random

class Markov:
    def __init__(self, n=1, delimiter=None, end=None, language='Japanese'):
        self.n = n
        if delimiter is not None:
            self.splitter = lambda text : text.split(delimiter)
        else: 
            if language.startswith('J'):
                self.splitter = lambda text : list(text)
            else:
                self.splitter = lambda text : text.split()
        if end is not None:
            self.end = end
        elif delimiter is not None:
            self.end = delimiter
        elif language.startswith('J'):
            self.end = ''
        else:
            self.end = ' '
            
    def make_dictionary(self, texts):
        """ マルコフ連鎖の辞書を作る """
        x_chain = self.splitter(texts)
        key     = self.splitter(texts)
        self.markov = {}
        key = []    # キーはリストで操作して辞書アクセスの際にタプルにする
        for x in x_chain:
            # -- 1つ前のxでキーの準備ができていたら実行 --
            if len(key) == self.n:
                k = tuple(key)
                # 該キーに対応するものがなければ空のリストで用意
                if k not in self.markov:
                    self.markov[k] = []
                # 該キーのリスト中に対象がなかったら加える
                if x not in self.markov[k]:
                    self.markov[k].append(x)
            # -- 次のxに向けてキーを整える --            
            key.append(x) 
            if len(key) > self.n:
                key = key[1:] # 左端を捨てて左シフト
        return self.markov    
      
    def generate_text(self, length=100, start=None):
        """ マルコフ連鎖の辞書を使って生成する """
        texts = start + self.end             # 文字列のまま扱う
        key = self.splitter(start)[-self.n:] # startを分割した末尾のn個
        #print('### key =', tuple(key))
        count = 0
        while count < length:
            # -- キーの指す候補からランダムに選んで綴っていく --
            options = self.markov.get(tuple(key), None) # 辞書のキーはタプル
            #print('###', options)
            if options is None:
                x = None
            else:    
                x = random.choice(options)
                texts += x + self.end
            count += 1
            # -- 次に向けてキーを左シフトする --    
            key.append(x)
            key = key[1:]

        return texts
     
if __name__=='__main__':
    text = ('This eBook is for the use of anyone anywhere in the United States '
    'and most other parts of the world at no cost and with almost '
    'no restrictions whatsoever. You may copy it, give it away or re-use it '
    'under the terms of the Project Gutenberg License included with this '
    'eBook or online at www.gutenberg.org. If you are not located in the '
    'United States, you will have to check the laws of the country where '
    'you are located before using this eBook.')

    markov = Markov(n=2, language='English')
    markov_dic = markov.make_dictionary(text)
    gen_text = markov.generate_text(100, 'If you')
    print(gen_text)

