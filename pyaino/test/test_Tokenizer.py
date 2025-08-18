from pyaino import common_function as cf


text = "こんにちは、日本。こんにちは、世界！"

options = (None,
           {'splitter':cf.split_by_janome},
           {'splitter':cf.split_japanese},
           {'unit':'語'},
           {'splitter':None, 'language':'Japanese', 'unit':None, 'delimiter':None, 'end':None}
         )

for o in options:
    print()
    if o is None:
        tokenizer = cf.Tokenizer(text) 
    else:
        tokenizer = cf.Tokenizer(text, **o)
    print('vocab_size =', tokenizer.vocab_size())
    token = tokenizer.encode(text)
    print(token)
    recoverd = tokenizer.decode(token)
    print(recoverd)


text = 'Hello Japan. Hello World!'

options = (None, # 文字分割なのでOK
           {'splitter':cf.split_english, 'language':'English'},
           {'splitter':cf.split_english, 'joiner':cf.join_english},
           {'language':'English'}, # 空白の挿入が必要　
           {'language':'English', 'unit':'語'}, # 空白の挿入が必要　
           {'language':'English', 'delimiter':' '}, # 空白の挿入が必要
           {'splitter':None, 'language':'English', 'unit':None, 'delimiter':None, 'end':None}
          )

for o in options:
    print()
    if o is None:
        tokenizer = cf.Tokenizer(text) 
    else:
        tokenizer = cf.Tokenizer(text, **o)
    print('vocab_size =', tokenizer.vocab_size())
    token = tokenizer.encode(text)
    print(token)
    recoverd = tokenizer.decode(token)
    print(recoverd)
