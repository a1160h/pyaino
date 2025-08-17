# common_function
# 2025.08.17 A.Inoue 

from pyaino.Config import *
from pyaino import Neuron as neuron
from pyaino import nucleus
from pyaino import LossFunctions as lf
import struct
import os
import sys
import gc
import re
import ast
import warnings
import matplotlib.pyplot as plt
import pickle
import inspect, types
try:
    from PIL import Image, ImageFilter
except:
    print('Please install Pillow.')

print('common_function np =', np.__name__)    

def is_user_defined_instance(obj):
    """ インスタンス化されたユーザ定義のクラスの判別(FunctionsやNeuronのクラス) """
    return not isinstance(obj, type) and type(obj).__module__ != 'builtins'

def get_object_by_id(target_id):
    """ gcを使ってidから元のオブジェクトを取得 """
    for obj in gc.get_objects():
        if id(obj) == target_id:
            return obj
    return None

def get_variable_name(obj):
    """ グローバルスコープ内での変数名を取得 """
    for name, value in list(globals().items()):
        if value is obj:
            return name
    return None

def get_argument_name(var):
    """ 関数の引数として渡されたオブジェクトの変数名を取得 """
    for name, value in inspect.currentframe().f_back.f_locals.items():
        if value is var:
            return name
    return None

def get_caller_var_name(var, depth=2, default='model'):
    """ 呼出し元におけるオブジェクトの変数名を呼出し先の中で取得 """
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            frame = frame.f_back
        callers_locals = frame.f_locals.items()
        names = [name for name, val in callers_locals if val is var]
        return names[0] if names else default
    finally:
        del frame

def member_in_module(module):
    """ モジュール内のメンバ名をリスト形式で返す """
    member = []
    for d in dir(module):
        if d.startswith('__') or d=='np': # 特殊メソッドやnpは読み飛ばす
            continue
        member.append(d)
    return member    

def extract_name_and_args(text):
    """ 'ClassName(...)' → ('ClassName', '...') """
    if '(' not in text:
        return text.strip(), ''
    name = text[:text.find('(')].strip()
    rest = text[text.find('(')+1:]
    depth = 1
    idx = 0
    for idx, c in enumerate(rest):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                break
    return name, rest[:idx]

def split_args(argstr):
    """ カンマで区切られた項目をネスト考慮して分割 """
    args = []
    current = ''
    depth = 0
    for c in argstr:
        if c == ',' and depth == 0:
            args.append(current.strip())
            current = ''
        else:
            current += c
            if c in '([{':
                depth += 1
            elif c in ')]}':
                depth -= 1
    if current:
        args.append(current.strip())
    return args

def parse_args_and_kwargs(argtext):
    """ '0.2, 0.3, beta=0.9' のような文字列を ([0.2, 0.3], {'beta': 0.9}) に分離 """
    args = []
    kwargs = {}
    for item in split_args(argtext):
        if '=' in item:
            key, value = item.split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                value = ast.literal_eval(value)
            except Exception:
                pass
            kwargs[key] = value
        else:
            try:
                val = ast.literal_eval(item)
            except Exception:
                val = item
            args.append(val)
    return args, kwargs

def who_called_me(depth=2):
    stack = inspect.stack()
    caller_frame = stack[depth] # 2:呼出し元->1:common_function->0:自分
    module = inspect.getmodule(caller_frame[0])
    return module


def who_am_i(target, stack_offset=1):
    """
    呼び出し元スコープ内で target に一致する変数を探し、
    target がその中の属性であれば '変数名.属性名...' のようなアクセスパスを返す．   
    frameを安全に扱う．frame.f_localsは呼び出し時点のローカルスナップショットのため要注意．
    """
    frame = inspect.currentframe()
    try:     # del frameを安全に
        for _ in range(stack_offset):
            frame = frame.f_back
        locals_dict = frame.f_locals.copy()  # 安全に評価するために.copy()
        path = search_obj_path(locals_dict, target)
        return path
    finally: # 必ず実行される
        del frame

def who_am_i_bkup(target, stack_offset=1):
    """
    呼び出し元スコープ内で target に一致する変数を探し、
    target がその中の属性であれば '変数名.属性名...' のようなアクセスパスを返す。
    """
    frame = inspect.currentframe()                 # この関数自身のframe
    for _ in range(stack_offset):                  # その外側へ
        frame = frame.f_back
    path = search_obj_path(frame.f_locals, target) # frameのローカル変数を探索
    del frame                                      # frame開放
    return path
    
def who_am_i2(target, stack_offset=1):
    """ 呼び出し元スコープ内で target を指す変数名（またはアクセスパス）を返す """
    frame = inspect.currentframe()
    try:
        for _ in range(stack_offset):
            frame = frame.f_back

        # スカラ一致を最優先
        for varname, val in frame.f_locals.items():
            if val is target:
                return varname

        # 一致しない場合のみ、再帰的に属性をたどる
        for varname, val in frame.f_locals.items():
            path = search_obj_path(val, target, name=varname, seen=set())
            if path:
                return path

        return None
    finally:
        del frame

def eval_in_module(class_name, module=None, *args, **kwargs):
    '''
    module内に定義されたクラスをclass_nameの文字列で指定してインスタンスを生成
    その際に、指定するクラスがmodule内に定義されているものかをチェックするとともに、
    class_nameの文字列中に記述した内容を引数に組み込む
    '''
    # class_nameからクラス名とオプションを抽出し、指定されたオプションで更新
    name, argtext = extract_name_and_args(class_name)
    parsed_args, parsed_kwargs = parse_args_and_kwargs(argtext)
    args += tuple(parsed_args)
    kwargs.update(parsed_kwargs)

    # 呼出し元のモジュールを設定
    if module is None:
        module = who_called_me()
    
    # モジュールに存在する定義の確認
    if not hasattr(module, name):
        raise Exception(f'Invalid class: {name} not in {dir(module)}')

    # nameの文字列を評価  
    function_class = getattr(module, name) # eval('module.'+name) 相当
    if not isinstance(function_class, type):
        raise TypeError(f"{name} is not a class.")

    # -- インスタンス化 --
    function = function_class(*args, **kwargs)
    return function
      
def eval_in_module_bkup(class_name, module, *args, **kwargs):
    '''
    module内に定義されたクラスをclass_nameの文字列で指定してインスタンスを生成
    evalの際にはmodule内に定義されているものかをチェックするとともに、
    インスタンス化したものには元のクラスの名前をnameとして付与する
    '''
    # -- moduleに定義されたクラス名のリストを作る --　
    dir_module = [] 
    for d in dir(module):
        if d.startswith('__') or d=='np': # 特殊メソッドやnpは読み飛ばす
            continue
        dir_module.append(d)
    #print('classes in module :', dir_module)
    
    # -- class_nameがモジュール内に定義されていない場合は例外発生 --
    if class_name not in dir_module: 
        raise Exception('Invalid function specified.', class_name,' Should be in', dir_module)

    # -- class_nameの文字列を評価してインスタンス化、元のclassから名前を継承 -- 
    function_class = eval('module.'+class_name)
    name = function_class.__name__   
    function = function_class(*args, **kwargs)
    setattr(function, 'name', name)  
    return function

# -- help --
def help(title):
    #fin = open('C:/Python37/Lib/site-packages/pyaino/help' + title, 'rt')
    path = os.path.dirname(os.path.abspath(__file__))
    fin = open(path + '/help' + title, 'rt')  
    comment = fin.read()
    fin.close()
    print(comment)

def get_entropy(x, bins=100, density=True, eps=1e-9):
    """ ヒストグラム推定によるエントロピーの算出 """
    hist, bin_edges = np.histogram(x, bins=bins, density=density)
    p = hist / (np.sum(hist) + eps)
    entropy = -np.sum(p * np.log(p + eps))
    return entropy

# -- オブジェクトの情報 --
def get_obj_info(obj, name=None, seen=None, generation=0):
    """オブジェクトとその内部要素を再帰的に表示"""
    #print('called', obj.__class__.__name__, name)#, len(seen), generation)
    #print('###', type(obj), name, obj.__class__.__name__, obj.__class__.__bases__)

    if name is None:   # はじめは呼び出し元でのobj名
        name = who_am_i(obj, stack_offset=2) # 0:who_am_i->1:ここ
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:  # すでに計算済みならスキップ
        return 
    seen.add(obj_id)

    # HDArrayやXArrayのアトリビュート(grad, creator, generation)は見ない
    if type(obj) in (nucleus.HDArray, nucleus.XArray):
        return
    
    # リスト、タプル、セット
    if isinstance(obj, (list, tuple, set)):
        for idx, item in enumerate(obj):
            get_obj_info(item, name=name+'['+str(idx)+']',
                         seen=seen, generation=generation)

    # 辞書型の場合（キーと値）
    elif isinstance(obj, dict):
        #generation += 1
        for key, value in obj.items():
            get_obj_info(value, name=key,
                         seen=seen, generation=generation)
            name = key
   
    # クラスのインスタンス（`__dict__` の中身）
    elif hasattr(obj, '__dict__'):
        print(' '*generation, generation, name, ':', obj.__class__.__name__)

        def print_info(item, label, generation):
            if isinstance(item, np.ndarray): # 20250530AI
                print(' '*(generation+3), label,
                  item.shape, 'dtype =', item.dtype, #'nbytes =', item.nbytes,
                  f'mean = {np.mean(item):5.3f} std = {np.std(item):5.3f}')
            else:
                print(' '*(generation+3), label, item)

        if not hasattr(obj, 'update'): # 20250406AI
            pass
        else:
            if hasattr(obj, 'w') and obj.w is not None:
                print_info(obj.w, 'weight', generation)
            if hasattr(obj, 'v') and obj.v is not None:
                print_info(obj.v, 'weight', generation)
            if hasattr(obj, 'b') and obj.b is not None:
                print_info(obj.b, 'bias', generation)
            if hasattr(obj, 'gamma') and obj.gamma is not None:
                print_info(obj.gamma, 'gamma', generation)
            if hasattr(obj, 'beta') and obj.beta is not None:
                print_info(obj.beta, 'beta', generation)
        generation = generation + 1
        name, seen, generation = get_obj_info(obj.__dict__, name=obj.__class__.__name__,
                                        seen=seen, generation=generation)

    return name, seen, generation

# -- オブジェクトのサイス --
def get_obj_size(obj, seen=None, verbose=False):
    """オブジェクトとその内部要素のメモリサイズを再帰的に計算"""
    
    # 計算済みのオブジェクトを記録する（循環参照対策）
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:  # すでに計算済みならスキップ
        return 0
    seen.add(obj_id)
    
    # 基本サイズ
    size = sys.getsizeof(obj)#; print('size of obj', size)

    # 辞書型の場合（キーと値のサイズも加算）
    if isinstance(obj, dict):
        #print('### dict ###')
        for key, value in obj.items():
            size += get_obj_size(key, seen, verbose)
            size += get_obj_size(value, seen, verbose)
    
    # リスト、タプル、セットなどの場合（要素のサイズを加算）
    elif isinstance(obj, (list, tuple, set)):
        #print('### tuple ###')
        for item in obj:
            size += get_obj_size(item, seen, verbose)

    elif isinstance(obj, np.ndarray):# 
        #print('### ndarray ###')
        array_size = obj.nbytes
        if verbose:
            print(obj.__class__.__name__, array_size)
        size += array_size 
    
    # クラスのインスタンス（`__dict__` の中身を加算）
    elif hasattr(obj, '__dict__'):
        #print('### class ###')
        class_size = get_obj_size(obj.__dict__, seen, verbose)
        if verbose:
            print(obj.__class__.__name__, class_size)
            #print(vars(obj).keys())
        size += class_size
    
    return size

def search_obj_path2(obj, target, name=None, seen=None):
    """ 対象objからtargetを再帰的に探索し、そのpathをnameと結合して返す """

    if seen is None:
        seen = set()
    if obj is None or not is_recursive_type(obj):
        return None
    obj_id = id(obj)
    if obj_id in seen:
        return None
    seen.add(obj_id)

    if obj is target:
        return name or "<?>" # nameがNoneなら"<?>"さもなくばname

    # HDArrayやXArrayのアトリビュート(grad, creator, generation)は見ない
    if type(obj) in (nucleus.HDArray, nucleus.XArray):
        return

    # リスト
    if isinstance(obj, list): 
        for i, item in enumerate(obj):
            path = search_obj_path(item, target, name=f"{name}[{i}]", seen=seen)
            if path:
                return path

    # 辞書
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = repr(key) if not isinstance(key, str) else key
            name_to = key_str if name is None else f"{name}.{key_str}"
            path = search_obj_path(value, target, name=name_to, seen=seen)
            if path:
                return path

    # 属性
    if hasattr(obj, '__dict__') and type(obj).__module__ != 'builtins':
        for attr, value in obj.__dict__.items():
            child = getattr(obj, attr, None)
            path = search_obj_path(child, target, name=f"{name}.{attr}" if name else attr, seen=seen)
            if path:
                return path
            
    return None

def search_obj_path(obj, target, name=None):
    """ 対象objからtargetを再帰的に探索し、そのpathをnameと結合して返す """
    if obj is target:
        return name or "<?>" # nameがNoneなら"<?>"さもなくばname
    
    # HDArrayやXArrayのアトリビュート(grad, creator, generation)は見ない
    if type(obj) in (nucleus.HDArray, nucleus.XArray):
        return

    # リスト
    if isinstance(obj, list): 
        for i, item in enumerate(obj):
            path = search_obj_path(item, target, name=f"{name}[{i}]")
            if path:
                return path

    # 辞書型
    if isinstance(obj, dict):
        for key, value in obj.items():
            name_to = key if name is None else f"{name}.{key}"
            path = search_obj_path(value, target, name=name_to)
            if path:
                return path

    # 属性
    if hasattr(obj, '__dict__') and type(obj).__module__ != 'builtins':
        for attr, value in obj.__dict__.items():
            child = getattr(obj, attr)
            path = search_obj_path(child, target, name=f"{name}.{attr}")
            if path:
                return path
            
    return None

def is_recursive_type(obj):
    """中を再帰的に探索してよい型だけ True を返す"""
    if obj is None:
        return False
    if isinstance(obj, (str, int, float, bool, complex)):
        return False
    if isinstance(obj, (type, types.FunctionType, types.BuiltinFunctionType,
                        types.MethodType, types.ModuleType)):
        return False
    if isinstance(obj, (np.ndarray, np.generic)):
        return False
    return isinstance(obj, (dict, list, tuple, set)) or hasattr(obj, '__dict__')

def search_obj_path3(obj, target, name=None, seen=set()):
    """ローカル変数名（name）から target へのパスを探索"""
    if id(obj) in seen or obj is None or not is_recursive_type(obj):
        return None
    seen.add(id(obj))

    if obj is target:
        return name

    # dict
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_str = repr(k) if not isinstance(k, str) else k
            path = search_obj_path(v, target, f"{name}.{k_str}", seen)
            if path:
                return path

    # list / tuple
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            path = search_obj_path(v, target, f"{name}[{i}]", seen)
            if path:
                return path

    # object attributes
    if hasattr(obj, '__dict__') and type(obj).__module__ != 'builtins':
        for attr, val in obj.__dict__.items():
            path = search_obj_path(val, target, f"{name}.{attr}", seen)
            if path:
                return path

    return None


def export_parameters_recursive(obj, params=None, name=None, param_label = ('w', 'v', 'b', 'gamma', 'beta'),
                             seen=None, generation=0):
    """オブジェクトとその内部要素を再帰的に表示"""
    if name is None:   # はじめは呼び出し元でのobj名
        name = who_am_i(obj, stack_offset=3) # 0:who_am_i->1:ここ->2:呼び出し元->3:__main__
    if seen is None:   # 明示的に初期化が必要
        seen = set()
    if params is None: # 明示的に初期化が必要
        params = {}
        
    obj_id = id(obj)

    if type(obj) in (type(None), int, float, bool, str):
        return params, name, seen, generation

    if obj_id in seen:  # すでに計算済みならスキップ
        return params, name, seen, generation
    seen.add(obj_id)

    # リスト、タプル、セット
    if isinstance(obj, (list, tuple, set)):
        for idx, item in enumerate(obj):
            # ndarrayに行きついたら、再帰呼び出しはしない
            if isinstance(item, np.ndarray):
                continue
            # 要素を対象に再帰呼び出し
            _, _, seen, _ = export_parameters_recursive(item, params=params,
                            name=f"{name}[{idx}]", seen=seen, generation=generation)

    # 辞書型の場合（キーと値）
    elif isinstance(obj, dict):
        for key, value in obj.items():
            # 辞書の値を対象に再帰呼び出し
            name_to = key if name is None else f"{name}.{key}"
            _, _, seen, _ = export_parameters_recursive(value, params=params,
                            name=name_to, seen=seen, generation=generation)
    
    # クラスのインスタンス（`__dict__` の中身）
    elif hasattr(obj, '__dict__'):
        if not hasattr(obj, 'update'): # 20250406AI
            pass
        elif obj.__class__.__base__.__name__ in ('OptimizerBase', 'ActivatorBase'):
            pass
        else:
            for p in param_label:
                if hasattr(obj, p):                  # pは文字列 
                    params[name+'.'+p] = getattr(obj, p) # 該当ケースに対象を辞書登録              

        generation = generation + 1
        # クラスに付随する辞書を対象に再帰呼び出し
        params, name, seen, generation = export_parameters_recursive(obj.__dict__, params=params,
                                         name=name, seen=seen, generation=generation)

    return params, name, seen, generation

def export_parameters(model, **kwargs): # targetを指定するなどのため
    """ モデルのパラメータを抽出する """
    params, name, seen, generation = export_parameters_recursive(model, **kwargs)
    return params

def get_param_info(model):
    params = export_parameters(model)
    for k, v in params.items():
        print(k, type(v), end='')
        if v is not None:
            print(v.shape, v.dtype)
        else:
            print()
    

def import_parameters_recursive(obj, params=None, name=None, param_label = ('w', 'v', 'b', 'gamma', 'beta'),
                             seen=None, generation=0):
    """ 再帰的にparamsをobjのtarget項に設定 """
    if name is None:   # はじめは呼び出し元でのobj名
        name = who_am_i(obj, stack_offset=3) # 0:who_am_i->1:ここ->2:呼び出し元->3:__main__
    if seen is None:   # 明示的に初期化が必要
        seen = set()
    if params is None: # 必ず外から与える
        raise Exception('No params provided.')

    obj_id = id(obj)

    if type(obj) in (type(None), int, float, bool, str):
        return name, seen, generation

    if obj_id in seen:  # すでに計算済みならスキップ
        return name, seen, generation
    
    seen.add(obj_id)

    # リスト、タプル、セット
    if isinstance(obj, (list, tuple, set)):
        #print('<<list, tuple, set>>')
        for idx, item in enumerate(obj):
            # ndarrayに行きついたら、再帰呼び出しはしない
            if isinstance(item, np.ndarray):
                continue
            # 要素を対象に再帰呼び出し
            _, seen, _ = import_parameters_recursive(item, params=params,
                         name=f"{name}[{idx}]", seen=seen, generation=generation)

    # 辞書型の場合（キーと値）
    elif isinstance(obj, dict):
        for key, value in obj.items():
            # 辞書の値を対象に再帰呼び出し
            name_to = key if name is None else f"{name}.{key}"
            _, seen, _ = import_parameters_recursive(value, params=params,
                         name=name_to, seen=seen, generation=generation)
            
    # クラスのインスタンス（`__dict__` の中身）
    elif hasattr(obj, '__dict__'):
        if not hasattr(obj, 'update'): # 20250406AI
            pass
        elif obj.__class__.__base__.__name__ in ('OptimizerBase', 'ActivatorBase'):
            pass
        else:
            for p in param_label:
                fullname = name+'.'+p
                if hasattr(obj, p):
                    if fullname in params: # paramsに無いものは放置20250725AI
                        setattr(obj, p, params[fullname])
                    else:
                        warnings.warn(fullname+' skipped.')

        generation = generation + 1
        # クラスに付随する辞書を対象に再帰呼び出し
        name, seen, generation = import_parameters_recursive(obj.__dict__, params=params,
                                 name=name, seen=seen, generation=generation)

    return name, seen, generation

def import_parameters(model, params, **kwargs): # targetを指定するなどのため
    name, seen, generation = import_parameters_recursive(model, params, **kwargs)


def analize_parameters(model):
    """ モデルのパラメータの統計情報 """
    params = export_parameters(model)
    statistics = {}
    for key, value in params.items():
        if value is None:
            shape, mu, sigma, sadicarnot = None, None, None, None
        else:
            shape = value.shape
            mu = np.mean(value)
            sigma = np.std(value)
            sadicarnot = get_entropy(value)
        print(key, 'shape =', shape, '\n',
              'mean =', mu, 'std =', sigma, 'entropy =', sadicarnot)
        statistics[key] = mu, sigma, sadicarnot
    return statistics    

def c2n_dictionary(parameters):
    """ 辞書に登録されたcupyの値をnumpyに変換する """
    print('c2n_dictionary called on np =', np.__name__)
    def conv(x):
        if np.__name__=='cupy': # cupy環境
            if isinstance(x, np.ndarray): # 現npに合致=cupy配列
                x = np.asnumpy(x)
            else:                         # 現npとは違う=numpy配列
                pass
        else:                   # numpy環境
            if isinstance(x, np.ndarray): # 現npに合致=numpy配列　
                pass
            elif x is not None:           # 現npとは違うcupy配列
                x = np.array(x.tolist(), dtype=Config.dtype)
        return x        
    for k, p in parameters.items():
        parameters[k] = conv(p)
    return parameters    

def n2c_dictionary(parameters):
    """ 辞書に登録されたnumpyの値をcupyに変換する """
    print('n2c_dictionary called on np =', np.__name__)
    def conv(x):
        if np.__name__=='cupy': # cupy環境
            if isinstance(x, np.ndarray): # 現npに合致=cupy配列
                pass
            elif x is not None:           # 現npとは違う=numpy配列
                x = np.array(x.tolist(), dtype=Config.dtype)
        else:                   # numpy環境
            pass # numpy環境では何もしない
        return x        
    for k, p in parameters.items():
        parameters[k] = conv(p)
    return parameters    


# -- 学習結果の保存(辞書形式) --
def save_parameters(file_name, model, numpy=True):
    params = export_parameters(model)
    if numpy:
        params = c2n_dictionary(params)
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)
    print(model.__class__.__name__, 'モデルのパラメータをファイルに記録しました=>', file_name)    

# -- 学習結果の継承(辞書形式) --
def load_parameters(file_name, model):
    with open(file_name, 'rb') as f:
        params = pickle.load(f)
    print('load_parameters called on np =', np.__name__)    
    if np.__name__=='cupy':
        params = n2c_dictionary(params)
    if np.__name__=='numpy':
        params = c2n_dictionary(params)
    import_parameters(model, params)
    print(model.__class__.__name__, 'モデルのパラメータをファイルから取得しました<=', file_name)
    return params
    
# -- 学習結果の保存(辞書形式) --
def save_parameters_cpb(file_name, title, params):
    """ 旧版互換性のため """
    # params は辞書形式(export_params で取得したもの)
    # 保存は常に numpy とする
    # 一般的に、numpy環境ではfloat32の方がfloat16よりも高速
    # 一方、cupy環境ではfloat16の方がfloat32よりも高速
    if np.__name__ =='cupy': # cupy環境
        for key in params.keys(): 
            params[key] = np.asnumpy(params[key]).astype('f2')
    else:                    # numpy環境
        for key in params.keys():
            params[key] = params[key].astype('f4')
    params['title'] = title 
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)
    print(title, 'モデルのパラメータをファイルに記録しました=>', file_name)    

# -- 学習結果の継承(辞書形式) --
def load_parameters_cpb(file_name):
    """ 旧版互換性のため """
    with open(file_name, 'rb') as f:
        params = pickle.load(f)
    title = params.pop('title', None)
    print(title, 'モデルのパラメータをファイルから取得しました<=', file_name)
    return title, params
    

# -- 標準化 --
class Normalize:
    def __init__(self, method='standard'):
        self.method = method
        self.shift = None
        self.base = None

    def __call__(self, data, adapt=True):
        return self.normalize(data, adapt=adapt)

    def normalize(self, data, adapt=True):
        method = self.method
        data = np.array(data)
        if method is None: # Noneでそのまま通過
            shift = 0
            base  = 1
        elif not method:   # Falseでそのまま通過
            shift = 0
            base  = 1
        elif method in('0to1', 'minmax01', 'range01'):
            data_min = np.min(data); data_max = np.max(data)
            shift = data_min
            base  = data_max - data_min
            print('データは最小値=0,最大値=1に標準化されます')
        elif method in('-1to1', 'minmax-11','range-1to1'):
            data_min = np.min(data); data_max = np.max(data)
            shift = (data_max + data_min)/2
            base  = (data_max - data_min)/2
            print('データは最小値=-1,最大値=1に標準化されます')
        elif method in('l2', 'l2n','norm', 'l2norm'):
            l2n = np.sum(data**2)**0.5
            shift = 0
            base  = l2n
            print('データはl2ノルムが1になるように標準化されます')
        else: # standard
            shift = np.average(data)
            base  = np.std(data)
            print('データは平均値=0,標準偏差=1に標準化されます')
        if adapt:
            self.shift = shift
            self.base  = base
        elif self.shift is None or self.base is None:
            raise Exception('Adaptation of data is needed.')
        data = (data - self.shift) / self.base
        return data       
 
    def denormalize(self, data):
        method = self.method
        data = np.array(data)
        data = data * self.base + self.shift
        return data       

def normalize(data, method='standard'):
    return Normalize(method)(data)

class NormalizeMulti:
    """ 列方向に複数項目が並んだデータを、それぞれ行方向で正規化する """
    def __init__(self, n_data, method='standard'):
        self.n_data = n_data
        self.method = method
        self.normalize = []
        for i in range(self.n_data):
            self.normalize.append(Normalize(self.method))
       
    def __call__(self, data, adapt=True):
        data_stack = []
        for i in range(self.n_data): # データの項目ごとに処理してまとめる
            data_i = data[:,i] 
            data_i = self.normalize[i](data_i, adapt=adapt)
            data_stack.append(data_i)
        data = np.array(data_stack, dtype='f4').T # 縦に積んだものを元の列方向に
        return data

    def denormalize(self, data):
        data_stack = []
        for i in range(self.n_data): # データの項目ごとに処理してまとめる
            data_i = data[:,i]
            data_i = self.normalize[i].denormalize(data_i)
            data_stack.append(data_i)
        data = np.array(data_stack, dtype='f4').T # 縦に積んだものを元の列方向に
        return data

def normalize_multi(data, method='standard'):
    return NormalizeMulti(method)(data)

class L2Normalize:
    def __init__(self):
        # 逆伝播なのか逆変換なのか怪しい        
        print("Normalizeをmethod='l2n'等を指定して使うか、または、" + \
              "Functionsに定義されたものを使ってください.")
        
    def normalize(self, x):
        x = np.array(x)
        l2n = np.sum(x**2, axis=-1, keepdims=True)**0.5
        y = x / l2n
        self.x = x
        self.l2n = l2n
        return y
   
    def denormalize(self, dy=1):
        x = self.x
        l2n = self.l2n
        dx = dy * (1 - x * x.sum(axis=-1, keepdims=True) / l2n**2) / l2n
        return dx

def l2normalize(data):
    return L2Normalize().normalize(data)


def cos_similarity(x, y, eps=1e-8):
    """ コサイン類似度 """
    x_hat = x / (np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True)) + eps)
    y_hat = y / (np.sqrt(np.sum(y ** 2, axis=-1, keepdims=True)) + eps)
    return np.dot(x_hat, y_hat.T)

# -- 白黒画像に変換 --
def rgb_to_gray(img, keepdims=True):
    if np.__name__=='cupy':
        arr = np.asnumpy(img.astype('uint8'))
    else:
        arr = img.astype('unit8')
    gray_data = []
    for i in range(len(img)):
        img_rgb = Image.fromarray(arr[i])
        img_gray = img_rgb.convert("L")
        gray_data.append(np.asarray(img_gray))
    gray_data = np.array(gray_data)
    if keepdims: # rgbの各チャネルに同一データを並べる 
        gray_data = np.repeat(gray_data, 3, axis=2)
        gray_data = gray_data.reshape(*img.shape)
    return gray_data    

# -- サンプルの提示と順伝播の結果のテスト --
def test_sample(show, func, x, t, label_list=None, label_list2=None):
    '''show:画像表示の関数, func:順伝播の関数,  X:入力, t:正解'''
    print('\n-- テスト開始 --')
    print('データ範囲内の数字を入力、その他は終了')
    while True:
        try:
            i = int(input('\nテストしたいデータの番号'))
            sample_data    = x[i:i+1, :] # 次元を保存して１つ取り出す
            sample_id      = int(t.reshape(-1)[i])
            sample_label   = label_list[sample_id]  if label_list  is not None else sample_id
        except:
            print('-- テスト終了 --')
            break
        print(' -- 選択されたサンプルを表示します --')
        print('このサンプルの正解は =>', sample_label)
        if input('サンプルの画像を表示しますか？(y/n)') in ('y', 'Y'):
            show(sample_data, sample_label) # 画像の表示
        
        # サンプル表示の後にいったん問合せた方が何をやっているか分りやすい 
        if input('機械の判定を行いますか？(y/n)') not in ('y', 'Y'):
            continue
        
        # 順伝播して結果を表示　　　　
        try:
            y = func(sample_data)             # 順伝播(入力の次元保存)
        except:
            y = func(sample_data.reshape(-1)) # 順伝播(入力をベクトル化)

        print(y.shape, sample_data.shape)

        if y.shape==sample_data.shape:
            print('-- ニューラルネットワークの出力を表示します --')
            show(y, sample_label)

        else:    
            print('ニューラルネットワークの出力\n', y)
            estimation = int(np.argmax(np.array(y.tolist()))) # ニューラルネットワークの判定結果        
        
            # Noneの判定には == / != ではなく is / is not が良い(判定がうまくいく)
            if       label_list is     None and label_list2 is None:
                estimate_label = estimation            
            elif     label_list is     None and label_list2 is not None:
                estimate_label = label_list2[estimation]
            elif     label_list is not None and label_list2 is None:
                estimate_label = label_list[estimation]
            else : # label_list is not None and label_list2 is not None:
                estimate_label = label_list2[estimation]
            
            print('機械の判定は　　 　　=>', estimate_label)

def confident_map(func, x, t, item_list=None, batch_size=1000,
                  homogenize=True, log=True, cmap='Blues',
                  title='confident_map', xlabel='target', ylabel='result'):
    """ カテゴリ分類問題で自信のほどを図示 """
    # -- ミニバッチ処理で結果を得る --
    y = []
    for i in range(0, len(x), batch_size):
        yi = func(x[i:i+batch_size]) 
        y += yi.tolist() # batch_size未満の端数も結合
    y = np.array(y)
    cat_n = y.shape[-1] # カテゴリ数

    # -- 表示用データ作成 --
    sum_yt, allsum_yt = [], []
    for i in range(cat_n):
        yt = y * (t == i).reshape(-1, 1)
        sum_yt.append(np.sum(yt, axis=0))
        allsum_yt.append(np.sum(yt))
    sum_yt = np.array(sum_yt)       # 正解で分けたカテゴリ毎の総計
    allsum_yt = np.array(allsum_yt) # 正解で分けた総計　
    if homogenize:
        allave = np.sum(allsum_yt) / cat_n # 全体度数の平均  
        allsum_yt /= allave 
        sum_yt /= np.where(allsum_yt>0, allsum_yt, 1) # 母数の違いを均一化
    # -- 画像表示 --
    if log:
        sum_yt = np.log(sum_yt+1)
        cbar_label = 'log(frequency+1)'
    else:
        cbar_label = 'frequency'
    if item_list is not None:    
        plt.rcParams['figure.subplot.bottom'] = 0.23
        plt.xticks(list(range(cat_n)), item_list, rotation=90)
        plt.yticks(list(range(cat_n)), item_list)
    plt.imshow(sum_yt.T.tolist(), origin='lower', cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def split_train_test(*data, **kwargs):
    '''
    リストかタプルで与えたデータそれぞれをrateで指定した割合でtrainとtestに分割
    shuffle=Trueにすればランダムにtrainとtestに振分ける
    '''
    rate    = kwargs.pop('rate', None)     # 分割割合
    shuffle = kwargs.pop('shuffle', False) # 分割の際にデータをシャッフルするか否か
    seed    = kwargs.pop('seed', None)     # 乱数のシード値
    n_data  = len(data[0])                 # データ数(各データの長さ)
    
    if rate == None or rate == 0:
        return data
    elif rate<0 or rate>1:
        raise Exception('rate should be in range 0 to 1.')

    print('分割割合 =', rate, 'シャッフル =', shuffle)
   
    index = np.arange(n_data)
    if seed is not None:
        np.random.seed(seed)
    if shuffle:
        np.random.shuffle(index)

    n_test  = int(n_data * rate)
    n_train = n_data - n_test

    index_train = index[:n_train].tolist()
    index_test  = index[n_train:].tolist() 

    train, test = [], []
    for d in data:
        train.append(d[index_train])
        test.append (d[index_test])  

    return train + test # リストの結合

def get_accuracy(y, t, mchx=False):
    '''
    y:順伝播の結果と、対応する t:正解とを与え、分類の正解率を返す
    mchx=Trueでは正誤表も返す
    '''
    result = np.argmax(y, axis=-1)
    if t.shape==y.shape: # 正解がone_hotの場合
        correct = np.argmax(t, axis=-1)
    elif t.ndim < y.ndim:
        correct = t
    else:
        raise Exception('Wrong dimension of t')
    errata = result == correct
    size = y.size / y.shape[-1] # 時系列データ対応
    accuracy = float(np.sum(errata) / size)
    if mchx:
        return accuracy, errata
    else:
        return accuracy

class Mesurement:
    def __init__(self, model, get_acc=None):
        self.model = model
        if get_acc is not None:
            self.get_acc = get_acc
        else:
            self.get_acc = get_accuracy # get_accuracyはcommon_function内に定義
        self.error, self.accuracy = [], []
    
    def __call__(self, x, t, n=None):
        if n is None or n > len(x): # 指定しないか、データ数より大きな数を指定した場合        
            n = len(x)
            x_sample = x
            t_sample = t
        else:                       # nを指定したらその数そのものがn
            #if n > len(x):
            #    print('与えられた入力のサンプル数は指定したサンプル数より小さいです')
            #    n = len(x)
            index_rand = np.arange(len(x))
            np.random.shuffle(index_rand)    # 
            index_rand = index_rand[:n]      # n個だけindexを取出す
            #print(type(index_rand), type(x), type(c))
            x_sample = x[index_rand, :]
            t_sample = t[index_rand, :]

        y = self.model.forward(x_sample)
        l = self.model.loss_function.forward(y, t_sample)
        acc = self.get_acc(y, t_sample)
        self.error.append(float(l))
        self.accuracy.append(acc)
        return float(l), acc

    def progress(self):
        return self.error, self.accuracy

    def graph(self):
        graph_for_error(self.error, self.accuracy)

#def mesurement(func, x, t):
#    return Mesurement(func)(x, t)

# -- 順伝播して誤差を計測、nを指定すると一部サンプルで計測 mchxは正解不正解の配列を返すかどうか --
def mesurement(func, loss_f, x, c, n=None, mchx=False):
    #x = np.array(x)#.tolist()) # 環境に合わせてtype変換
    #c = np.array(c)#.tolist()) # 環境に合わせてtype変換　
    # x : 入力  c : 正解値(one hot)  x と c の一部または全部を抽出
    if n is None or n > len(x): # 指定しないか、データ数より大きな数を指定した場合        
        n = len(x)
        x_sample = x
        c_sample = c
    else:                       # nを指定したらその数そのものがn
        #if n > len(x):
        #    print('与えられた入力のサンプル数は指定したサンプル数より小さいです')
        #    n = len(x)
        index_rand = np.arange(len(x))
        np.random.shuffle(index_rand)    # 
        index_rand = index_rand[:n]      # n個だけindexを取出す
        #print(type(index_rand), type(x), type(c))
        x_sample = x[index_rand, :]
        c_sample = c[index_rand, :]

    # 正解がターゲットインデクスで与えられる場合には one-hot 表現に変換
    if len(c) == c.size: # c がone-hotでない場合
        t_sample = c_sample                          # t_sampleはターゲットid(非one-hot)
        cz = c_sample.reshape(-1)
        maxcz = max(cz); mincz = min(cz)
        size = int(maxcz - mincz + 1)
        c_sample = np.zeros((n, size))               # c_sampleは一旦初期化して
        for i in range(n):                           #        one-hotに置換える
            c_sample[i, cz[i]-mincz] = 1.0
    else:                # c がone-hotの場合
        t_sample = np.argmax(c_sample, axis=1)       # c_sampleはそのままでone-hot

    errorc = 0                                  # エラー集約
    rt = np.empty(n)                            # 結果表
    n2 = n if n < 200 else 100                  # n2 は n (n小) または 200(n大)　　 
    for i in range(0, n, n2):                   # n小では全体を1回で処理、n大では一部ずつ処理
        xi = x_sample[i:i+n2]
        ci = c_sample[i:i+n2]
        ti = t_sample[i:i+n2]
        yi = func(xi)                           # 順伝播
        rt[i:i+n2] = np.argmax(yi, axis=1)      # 結果ターゲットid
        error = loss_f.forward(yi, ci)
        errorc += error * n2
        #errorc += loss_function(loss_f, yi, ci) * n2
        
    errata = np.array(rt==t_sample, dtype=bool) # 正誤表(ブール値)
    correct_rate = float(np.sum(errata) / n)    # 正解率　　　　　　　　　
    error_rate   = float(errorc / n)            # エラー率
    if mchx: # 正誤表を要求された場合
        #errata = errata.tolist()                # numpy環境から呼ばれる場合のため
        #rt     = rt.tolist()                    # numpy環境から呼ばれる場合のため　　　
        rt = rt.astype(int)
        return error_rate, correct_rate, errata, rt
    else:
        return error_rate, correct_rate

# -- 結果を与えて誤差を計測、nを指定すると一部サンプルで計測 mchxは正解不正解の配列を返すかどうか --
def mesure_no_propagation(loss_f, y, c, mchx=False):
    #y = np.array(y.tolist()) # 環境に合わせてtype変換
    #c = np.array(c.tolist()) # 環境に合わせてtype変換　
    # 正解がターゲットインデクスで与えられる場合には one-hot 表現に変換 
    if len(c) == c.size: # c がone-hotでない場合
        t = c                             # tはターゲットid(非one-hot)
        cz = c.reshape(-1)
        maxcz = max(cz); mincz = min(cz)
        ci = np.zeros((n, maxcz - mincz + 1))  # c1は一旦初期化して
        for i in range(n):                     #        one-hotに置換える
            ci[i, cz[i]-mincz] = 1.0
    else:                # c がone-hotの場合
        ci = c
        t = np.argmax(c, axis=1)               # cはそのままでone-hot

    rt = np.argmax(y, axis=1)                  # 結果ターゲットid 
    errata = np.array(rt==t, dtype=bool)       # 正誤表(ブール値)
    correct_rate = float(np.sum(errata) / len(c)) # 正解率
    error_rate   = float(neuron.loss_function(loss_f).forward(y, ci)) 
    #error_rate   = float(loss_function(loss_f, y, ci)) 

    if mchx: # 正誤表を要求された場合　 
        #errata = errata.tolist()
        #rt     = rt.tolist()
        return error_rate, correct_rate, errata, rt
    else:
        return error_rate, correct_rate

class Mesurement_for_GAN:
    def __init__(self, model):
        self.model = model
        self.error, self.accuracy = [], []

    def get_accuracy(self, y, t):
        correct = np.sum(np.where(y<0.5, 0, 1) == t)
        return correct / len(y)

    def __call__(self, x, t): 
        y = self.model.dsc.forward(x)
        loss = self.model.loss_function.forward(y, t)
        acc  = self.get_accuracy(y, t)
        loss = float(loss)
        acc  = float(acc)
        self.error.append(loss)
        self.accuracy.append(acc)
        return loss, acc

    def progress(self, moving_average=None):
        if moving_average is None:
            return self.error, self.accuracy
        else:
            r = int(moving_average)
            n = len(self.error) - r
            err_ma = []
            acc_ma = []
            for i in range(n):
                err_ma.append(float(np.average(self.error[i:i+r])))
                acc_ma.append(float(np.average(self.accuracy[i:i+r])))
            return err_ma, acc_ma    


def moving_average(x, period=1, stride=1):
    """ 移動平均 """
    period = int(period) 
    y = []
    for i in range(0, len(x) - period + 1, stride):
        y.append(float(np.mean(x[i:i+period])))
    return y

def moving_average_multi(x, period=1, stride=1, dtype='f4'):
    """ 行列の列方向に複数項目が並んだデータを、それぞれ行方向で区間平均をサンプリングする """
    tail = len(x)-period+1
    y = [np.mean(x[i:i+period], axis=0, dtype=dtype) for i in range(0, tail, stride)]
    return np.array(y)#, dtype=dtype)

def moving_average_multi_bkup(x, period=1, stride=1, dtype='f4'):
    """ 列方向に複数項目が並んだデータを、それぞれ行方向で区間平均をサンプリングする """
    x_stack = []
    tail = len(x)-period+1
    for i in range(len(x[0])): # データの項目ごとに処理してまとめる
        xi = x[:,i]
        #xi = moving_average(xi, period, stride)
        xi = [float(np.mean(xi[i:i+period])) for i in range(0, tail, stride)]
        x_stack.append(xi)
    x = np.array(x_stack, dtype=dtype).T # 縦に積んだものを元の列方向に
    return x

def select_category(x, stocastic=False, beta=2):
    """ ダミー変数をカテゴリ変数に変換 """
    x = np.array(x).reshape(-1)
    if stocastic: # 確率的に
        p = np.empty_like(x, dtype='f4') # 極小値多数でエラーしないため精度が必要
        p[...] = x ** beta
        p = p / np.sum(p)
        y = np.random.choice(len(p), size=1, p=p)
    else:         # 確定的に
        y = np.argmax(x, keepdims=True)
    return y

## -- 損失関数 --
def loss_function_wrapped(func, loss_f, x, t):
    y = func(x)
    return neuron.loss_function(loss_f).forward(y, t)

## -- 損失関数 --
def loss_function(loss_f, y, t):
    print('Not recommended to use loss_function in', __name__)
    if   y.ndim == 3: # 時系列データの場合 (B, T, n)
        batch_size = y.shape[0] * y.shape[1]
    elif y.ndim == 2: # バッチ処理 (B, n)
        batch_size = y.shape[0]
    else:
        batch_size = 1
    #print(y.shape, t.shape)    
    if loss_f == 'x_entropy': # 交差エントロピー誤差
        error = -np.sum(t * np.log(y + 1e-7))  / batch_size  
    else:                     # 2乗和誤差
        error = 1/2 * np.sum(np.square(y - t)) / batch_size
    '''
    順伝播と逆伝播の論理的な一貫性を配慮するならば、損失関数は正則化項を勘案すべき。
    しかしながら、逆伝播で勾配を求める際のパラメタ(重み)との関連を考えるならば、
    どの層のパラメタ(重み)を評価に加えるのか釈然としない。
    すなわち、出力層の重みを評価に加えれば良いではないかという考え方はできるが、
    その場合にかくれ層も正則化項を加える一方で損失関数には直接寄与しないことが説明できない。
    また逆に、損失関数の算出の際に正則化項を無視したとしても、パラメタの調整には全く影響しない。
    そこで、neuronのクラスで、勾配算出の際に正則化項を加える一方で、
    損失関数は、正則化項を加えずに算出することとする。
    
    '''
    return float(error)

## -- 数値微分 --
def numerical_gradient(func, x, h=1e-4): # 数値微分、変数xは配列に対応
    if isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x, dtype=float)     # 変数xの型指定は必須(整数型だと+h, -hが上手くいかない) 
    x_shape = x.shape ; x_size = x.size  # 変数xの形状と大きさ
    x = x.reshape(-1)                    # 要素を順に操作するためにベクトル化
    grad = np.zeros_like(x, dtype=Config.dtype)
    for i in range(x_size):
        xp = x.copy()
        xn = x.copy()
        xp[i] += h
        xn[i] -= h
        grad[i] = (func(xp.reshape(x_shape)) - func(xn.reshape(x_shape))) / (2 * h)
    grad = grad.reshape(x_shape)         # gradをxの形状にして返す
    return grad

def numerical_gradient_bkup(func, xi):        # 数値微分、変数xは配列に対応　
    h = 1e-4
    x = np.array(xi, dtype=float)        # 変数xの型指定は必須(整数型だと+h, -hが上手くいかない) 
    x_shape = x.shape ; x_size = x.size  # 変数xの形状と大きさ
    x = x.reshape(-1)                    # 要素を順に操作するためにベクトル化
    tmp  = np.zeros(x_size)
    grad = np.zeros(x_size)
    for i in range(x_size):              # xの要素を順に選んで±h
        tmp  = x[i]                       
        x[i] = tmp + h
        fxph = func(x.reshape(x_shape))  # 要素x[i]を+hしたx(他は元の値)で計算
        x[i] = tmp - h
        fxmh = func(x.reshape(x_shape))  # 要素x[i]を-hしたx(他は元の値)で計算   
        grad[i] = (fxph - fxmh) / (2 * h)  
        x[i] = tmp                       # x[i]をもとに戻す 
    grad = grad.reshape(x_shape)         # gradをxの形状にして返す
    return grad

def numerical_gradient2(func, x, h=1e-4):
    return (func(x+h) - func(x-h)) / (2*h)

## -- 勾配チェック -- 
def gradient_check(grad1, grad2):
    result =''
    for key in grad1.keys():
        grad1av = np.average(np.abs(grad1[key]))
        grad2av = np.average(np.abs(grad2[key]))
        diff = np.average(np.abs(grad1[key]-grad2[key]))
        #diff = np.average(grad2[key]/grad1[key])
        result += key + ':' +   '{:.2e} '.format(grad1av) \
                            +   '{:.2e} '.format(grad2av) \
                            + '△{:.2e} '.format(diff)
    return result



# -- 誤差の記録をグラフ表示 --
def graph_for_error(*data, **kwargs):
    labels = kwargs.pop('label',  None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    xlim   = kwargs.pop('xlim',   None)
    ylim   = kwargs.pop('ylim',   None)
    legend = True
    
    if labels is None: # labelsがない場合はNoneをdata分並べる
        labels = (None,) * len(data)
        legend = False

    elif len(data)==1 and type(labels) is str:
        labels = labels,

    elif type(labels) in(list, tuple) and len(data)==len(labels):
        pass
    
    else:
        raise Exception('length of data and label mismatch.')
       
    for d, l in zip(data, labels):
        plt.plot(d, label=l)
    if legend:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    fig = plt.gcf()
    plt.show()
    return fig

# -- 誤差の記録をグラフ表示 --
def graph_for_error2(data1, data2=None, labels=None):
    if type(data1)==tuple:
        data = []
        for d in data1:
            data.append(np.array(d).tolist()) 
        data1 = data    

    if type(data2)==tuple:
        data = []
        for d in data2:
            data.append(np.array(d).tolist()) 
        data2 = data    

    if data2 is not None:
        x_data = data1
        y_data = data2
    else:
        x_data = None  
        y_data = data1

    label = ['Train error', 'Train accuracy', 'Test error', 'Test accuracy']
    if labels is not None:
        for i , l in enumerate(labels):
            label[i] = l
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1) #fig.add_subplot(行,列,場所)

    if len(y_data) == 4:
        x_data = range(len(y_data[0])) if x_data is None else x_data
        ax2 = ax1.twinx()
        ax1.plot(x_data, y_data[0], color='blue',  linestyle = 'solid',  label=label[0])
        ax1.plot(x_data, y_data[2], color='teal',  linestyle = 'solid',  label=label[2])
        ax2.plot(x_data, y_data[1], color='red',   linestyle = 'dashed', label=label[1])
        ax2.plot(x_data, y_data[3], color='coral', linestyle = 'dashed', label=label[3])
        ax1.legend(bbox_to_anchor=(0.5, 1.15))
        ax2.legend(bbox_to_anchor=(1.0, 1.15))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Error')
        ax2.set_ylabel('Accuracy')
        plt.show()

    elif len(y_data) == 3:
        x_data = range(len(y_data[0])) if x_data is None else x_data
        ax1.plot(x_data, y_data[0], color='blue',  linestyle = 'solid',  label=label[0])
        ax1.plot(x_data, y_data[1], color='coral', linestyle = 'solid',  label=label[2])
        ax1.plot(x_data, y_data[2], color='teal',  linestyle = 'dashed', label=label[1])
        ax1.legend(bbox_to_anchor=(0.5, 1.15))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Error')
        plt.show()

    elif len(y_data) == 2:
        x_data = range(len(y_data[0])) if x_data is None else x_data
        ax1.plot(x_data, y_data[0], color='blue',  linestyle = 'solid',  label=label[0])
        ax1.plot(x_data, y_data[1], color='teal',  linestyle = 'solid',  label=label[2])
        ax1.legend(bbox_to_anchor=(0.5, 1.15))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Error')
        plt.show()

    else:
        x_data = range(len(y_data))    if x_data is None else x_data
        ax1.plot(x_data, y_data,   color='blue',   linestyle = 'solid',  label=label[0])
        ax1.legend(bbox_to_anchor=(0.5, 1.15))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Error')
        plt.show()

# -- 数値 n について、区間 c の間連続する並びを作り、残りの端数は末尾につける
def intermittent_random(n, c):
    i  = n // c; j = n % c
    irn = np.arange(i*c).reshape(i, c)
    idx = np.arange(i)
    np.random.shuffle(idx)
    irn = irn[idx]
    irn = np.concatenate([irn.reshape(-1), np.arange(i*c, i*c+j)])
    return irn

# -- インデクス(番号)で与えられるデータを one_hot形式に変換 --
def convert_one_hot(x, width): # x は入力、widthは one hot の幅(例えば0,1,2,3ならばwidth=4)　
    x = np.array(x)
    x_shape = x.shape
    x = x.reshape(-1)
    N = len(x)
    one_hot = np.empty((N, width), dtype=bool)
    for i, target in enumerate(x):
        one_hot[i][...]    = 0
        one_hot[i][target] = 1
    one_hot_shape = x_shape + (width,) # タプルの結合
    one_hot = one_hot.reshape(one_hot_shape)    
    #print('形状{} == 変換 ==> 形状{}'.format(x_shape, one_hot_shape))
    return one_hot

def split_english(text):
    text = text.lower()
    text = text.replace('o.k.', 'okay.') # ただ'ok'はうまくいかない
    text = text.replace('okey', 'okay ')
    text = text.replace('_', '_ ' ) # '_'は起動マークとして使用するから切断必須20220203    
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    text = text.replace('?', ' ? ')
    text = text.replace('!', ' ! ')
    text = text.replace('"', ' " ')
    text = text.replace('(', ' ( ')
    text = text.replace(')', ' ) ')
    text = text.replace('\n', ' \n ')
    text = text.replace("'m", " am")
    text = text.replace("'ve", " have")
    text = text.replace("'ll", " will")
    text = text.replace("'d", " would")
    text = text.replace("'re", " are")
    text = text.replace(" can't", " can not")
    text = text.replace(" won't", " will not")
    text = text.replace("n't", " not")
    text = text.replace("let's", "let us")
    text = text.replace("it's", "it is")
    text = text.replace("that's", "that is")
    text = text.replace("what's", "what is")
    text = text.replace("there's", "there is")
    text = text.replace("here's", "here is")
    text = text.replace("he's", "he is")
    text = text.replace("she's", "she is")
    text = text.replace("'", " ' ")
    text = text.replace('  ', ' ')

    return text.split(' ') if len(text)>1 else text

# -- 英語文章を idから辞書で語に変換しつつ整形して印字 ----
def join_english(data, print_out=False):
    last_post = 'upper'; last_w = ''; text = ''
    for w in data:
        post = None
        sep = ' '
        if w == last_w:
            pw = ''
        elif w == '<eos>':
            pw = '.'
            post = 'upper'
        elif w == '\n':
            pw = w
            post = 'upper'
        elif w == '.':
            pw = '.'
            post = 'upper' 
        elif w == '?':
            pw = '?'
            post = 'upper' 
        elif w == '!':
            pw = '!'
            post = 'upper' 
        elif w == ',':
            pw = ','
            sep = ''
        elif w == 's':
            pw = 's'
            sep = ''
        elif w == 't':
            pw = 't'
            sep = ''
        elif w == 'd':
            pw = 'd'
            sep = ''
        elif w == 've':
            pw = 've'
            sep = ''
        elif w == 'll':
            pw = 'll'
            sep = ''
        elif w == "'s":
            pw = "'s"
        elif w == "'":
            pw = "'"
            sep = ''
        elif w == "n't":
            pw = "n't"
        elif w == '<unk>':
            pw = ' XXX'
        elif w == 'N':
            pw = ' ###'
        elif w == 'i':
            pw = ' I'
        elif last_post == 'upper':
            pw = sep + w.capitalize()
            sep = ' '
        else:
            pw = sep + w
            sep = ' '
        text += pw
        if print_out:    
            print(pw, end ='')
        last_post = post
        last_w = w

    if print_out:     
        print('\n')
    return text

# -- 英語文章を idから辞書で語に変換しつつ整形して印字 ----
def print_en_texts_bkup(data, id_to_word={}):
    last_post = 'upper'; last_w = ''
    for d in data:
        post = None
        sep = ' '
        w = id_to_word[int(d)]
        if w == last_w:
            pw = ''
        elif w == '<eos>':
            pw = '.'
            post = 'upper'
        elif w == '\n':
            pw = w
            post = 'upper'
        elif w == '.':
            pw = '.'
            post = 'upper' 
        elif w == '?':
            pw = '?'
            post = 'upper' 
        elif w == '!':
            pw = '!'
            post = 'upper' 
        elif w == ',':
            pw = ','
            sep = ''
        elif w == 's':
            pw = 's'
            sep = ''
        elif w == 't':
            pw = 't'
            sep = ''
        elif w == 'd':
            pw = 'd'
            sep = ''
        elif w == 've':
            pw = 've'
            sep = ''
        elif w == 'll':
            pw = 'll'
            sep = ''
        elif w == "'s":
            pw = "'s"
        elif w == "'":
            pw = "'"
            sep = ''
        elif w == "n't":
            pw = "n't"
        elif w == '<unk>':
            pw = ' XXX'
        elif w == 'N':
            pw = ' ###'
        elif w == 'i':
            pw = ' I'
        elif last_post == 'upper':
            pw = sep + w.capitalize()
            sep = ' '
        else:
            pw = sep + w
            sep = ' '
        print(pw, end ='')
        last_post = post
        last_w = w
        #sep = ' '
    print('\n')
    return None

# -- 英語文章を idから辞書で語に変換しつつ整形して印字 ----
def print_en_texts(data, id_to_word={}):
    text = [id_to_word[int(d)] for d in data]
    join_english(text, print_out=True)
    return None

# -- 英語文章を単語単位で corpus と辞書に変換 ---------
#    既存のcorpusと辞書を渡すと辞書に追加して corpus を拡張
def preprocess_en(text, corpus=[], word_to_id={}, id_to_word={}):
    corpus = corpus.tolist() if type(corpus)==np.ndarray else corpus
    #corpus=np.array(corpus).tolist()     # リスト,arrayの両方に対応
    char_list = sorted(list(set(text)))  # 文字リスト作成、setで文字の重複をなくす
    #print('テキストの文字数:', len(text)) # len() で文字列の文字数を取得
    #print('文字数（重複無し）:', len(char_list))

    words = split_english(text)
    
    # word_to_id, id_to_word の生成
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # corpusへの変換
    for word in words:
        corpus.append(word_to_id[word])
    #corpus = np.array([word_to_id[word] for word in text]) # 追加はできないからNG   
    return corpus, word_to_id, id_to_word    

# -- 日本語文章を文字単位で corpus と辞書に変換 ---------
#    既存のcorpusと辞書を渡すと辞書に追加して corpus を拡張
def preprocess_jp(text, corpus=[], char_to_id={}, id_to_char={}):
    corpus = corpus.tolist() if type(corpus)==np.ndarray else corpus
    #corpus=np.array(corpus).tolist()     # リスト,arrayの両方に対応
    char_list = sorted(list(set(text)))  # 文字リスト作成、setで文字の重複をなくす
    #print('テキストの文字数:', len(text))          # len() で文字列の文字数を取得
    #print('文字数（重複無し）:', len(char_list))

    for char in text:
    #for char in char_list:     
        if char not in char_to_id:
            new_id = len(char_to_id)
            char_to_id[char] = new_id
            id_to_char[new_id] = char

    # corpusへの変換
    for char in text:
        corpus.append(char_to_id[char])
    #corpus = np.array([char_to_id[char] for char in text]) # 追加はできないからNG   
    return corpus, char_to_id, id_to_char    

def preprocess(*args, **kwargs):
    return preprocess_jp(*args, **kwargs)

def appearance_frequency(x):
    """ x(リスト形式)に出現する事象idを数えて辞書形式で返す """
    # <注>:xにcorpusを入力すれば、id_to_wordに対応する頻度が得られる
    counter = {}
    for i in x:
        if i not in counter:
            counter[i]  = 1
        else:
            counter[i] += 1
    return counter        

def split_by_janome(text):
    """ janomeの形態素分析で文章を語に分割 """
    from janome.tokenizer import Tokenizer
    word_list = Tokenizer().tokenize(text, wakati=True)
    return list(word_list)
    
def split_japanese(text):
    """ 日本語の文章を簡易的に語に分割 """
    import re
    tokens = re.findall(r'[一-龯]+|[ぁ-ん]+|[ァ-ヴー]+|[a-zA-Z0-9]+|[^\w\s]', text)
    return tokens
    
def print_jp_texts(data, id_to_char={}):
    for d in data:
        print(id_to_char[int(d)], end='')
    print('\n')    


# -- テキストデータの加工 ----
#    読み飛ばす行の指定、開始位置の文字の指定、改行ないし終了の指定
#    これらの指定にしたがって文字列を加工する
#    なお、開始文字を指定した場合、次の開始位置までは一つながりにする
def shape_text_data(texts, **kwargs):
    away  = kwargs.pop('away',   None) # 読み飛ばす指定
    drop  = kwargs.pop('drop',   None) # 削除する指定　
    start = kwargs.pop('start',    '') # 新たなテキストの開始
    end   = kwargs.pop('end',    '\n') # 改行ないし終了
    atach_limit = kwargs.pop('limit', 1000) # 行の結合をやめる長さ

    temp_line, good_line = '', ''
    texts_shaped =[]
    for i, line in enumerate(texts):
        if drop is not None:
            line = line.replace(drop, '')
        if line[0] in away:   # 無視する行の指定
            line = ''         # 空行にする
            good_line = temp_line + end
            temp_line = ''
        m = line.find(start)  # 見つからないときは-1
        line = line.replace(end, '')
        if m == -1 and len(temp_line) < atach_limit:
            temp_line += line # 抽出済みの行に新しい行を結合　
        if m >= 0 or i==(len(texts)-1):
            good_line = temp_line + end
            temp_line = line[m+1:] if m>0 else line
        if len(good_line) > 1:
            texts_shaped.append(good_line) 
            good_line = ''
    return texts_shaped

class Tokenizer:
    """ 日本語、英語の両方に対応するTokenizer """

    def __init__(self, text=None, splitter=None, joiner=None, default=None, 
                 language='Japanese', unit=None, delimiter=None, end=None):
        print(self.__class__.__name__, splitter, joiner, language, unit, delimiter, end)

        # splitterを定義
        if splitter is not None and isinstance(splitter, types.FunctionType):
            self.splitter = splitter
            print('splitter is', splitter.__name__)
            btwo = '' if language.startswith(('J', 'j')) else ' '
        elif delimiter is not None: # 区切り文字を指定
            self.splitter = lambda text : text.split(delimiter)
            print('splitter is according to specified delimiter.')
            btwo = delimiter
        elif unit in ('語', 'W', 'w', 'word') and language.startswith(('J', 'j')): # 日本語語分割　
            self.splitter = split_japanese
            print('splitter is a simple one based on regular expressions')
            btwo = ''
        elif unit in ('語', 'W', 'w', 'word'): # 英語など語分割
            self.splitter = lambda text : text.split()
            print('splitter is the built-in split() function in Python.')
            btwo = ' '
        else: # 文字分割
            self.splitter = lambda text : list(text)
            print('splitter is character based.')
            btwo = ''
        self.end = btwo if end is None else end

        # joinerを定義
        if joiner is not None and isinstance(joiner, types.FunctionType):
            self.joiner = joiner
            print('joiner is', joiner.__name__)
        else:
            self.joiner = lambda data : self.end.join(data)
            print('joiner is built-in join() function in Python.')

        if text is None:
            data = None
        elif isinstance(text, (list, tuple)):
            data = text
        else:
            data = self.splitter(text)

        self.create_vocab(data, default)

    def vocab_size(self):
        return len(self.token2id)
     
    def create_vocab(self, data=None, default=None):
        """ リスト形式のdataからtokenとidの間の双方向の変換の辞書を作る """
        # defaultに応じたtoken2idとid2tokenの初期化
        if default is None:
            self.token2id = {}
        elif type(default)==dict:
            self.token2id = default
            self.token2id.pop('<unk>', None) # <unk>は一旦削除
        self.id2token = {v: k for k, v in self.token2id.items()}
        used_ids = set(i for i in self.id2token) # 使用idの集合

        if data is not None: # dataから辞書を作る
            new_id = 0 # forループの中では0に初期化する必要はなく続きから探せば良い
            for token in data:
                if token in self.token2id:
                    continue
                # 使用されていないidを順に探す
                #new_id = 0
                while new_id in used_ids:
                    new_id += 1
                # 見つかった欠番のidで変換辞書を登録し、使用idの集合に追加    
                self.token2id[token]  = new_id
                self.id2token[new_id] = token
                used_ids.add(new_id)
            
        # 末尾に<unk>を加える        
        last_id = len(self.token2id) 
        self.token2id["<unk>"] = last_id
        self.id2token[last_id] = "<unk>"
        return

    def encode(self, text, ndarray=True):
        """ textを分割してidに変換し、それをindicesで返す """
        data = self.splitter(text)
        unk_id = self.token2id["<unk>"]
        indices = [self.token2id.get(d, unk_id) for d in data]
        if ndarray:
            indices = np.array(indices)
        return indices

    def decode(self, indices):
        """ indicesからtokenに変換して結合し文章textを返す """
        if isinstance(indices, np.ndarray):  
            indices = indices.tolist()
        elif type(indices)==int:
            indices = [indices]    
        data = [self.id2token.get(idx, "<unk>") for idx in indices]
        text = self.joiner(data)
        return text

    # -- 学習結果の保存(辞書形式) --
    def save(self, file_name):
        params = self.token2id
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        print(self.__class__.__name__, '辞書をファイルに記録しました=>', file_name)    

    # -- 学習結果の継承(辞書形式) --
    def load(self, file_name):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        self.create_vocab(default=params)
        print(self.__class__.__name__, '辞書をファイルから取得しました<=', file_name)
        

def arrange_mini_batch(data, batch_size=100, time_size=35, step=None, CPT=None):
    """ ひと続きの時系列データをミニバッチ処理に適した形に加工する """
    step = time_size if step is None else step
    if isinstance(data, (list, tuple)):
        dtype = type(data[0])
        data = np.array(data, dtype=dtype) # データの型は要素の型を継承
    data_size = len(data)
    vector_length = 1 if data.ndim==1 else data.shape[-1]
    print('データは', data_size, '個、その個々のデータのベクトル長は', vector_length)
    cut_length = data_size -1 -step*(batch_size -1)
    iters = cut_length//time_size       # 展開数は整数
    cut_length = iters * time_size
    if cut_length <= 0:
        raise Exception('データ長に対してステップ×バッチサイズが大きすぎます')
    print(cut_length)
    print('データから有効長', cut_length, 'の長さで', step, 'ずつずらして',
          batch_size, '回切出します')
    d = []
    for b in range(batch_size):
        d.append(data[b*step:b*step+cut_length+1]) # バッチ方向にリストで積んでいく
    d = np.array(d, dtype=data[0].dtype) # データの型を継承しつつ全体をndarrayに　
    xs, ts = d[:, :-1], d[:, 1:]         # ひとつずらして切り出す
    if vector_length==1:
        xs = xs.reshape(batch_size, -1, time_size).transpose(1, 0, 2)
        ts = ts.reshape(batch_size, -1, time_size).transpose(1, 0, 2)
        if CPT is not None:
            ts = ts[:,:, -CPT:]          # 時系列の末尾からCPTだけ遡ったところから末尾まで
        if CPT==1:
            ts = ts.reshape(iters, batch_size)
    else:
        xs = xs.reshape(batch_size, -1, time_size, vector_length).transpose(1, 0, 2, 3)
        ts = ts.reshape(batch_size, -1, time_size, vector_length).transpose(1, 0, 2, 3)
        if CPT is not None:
            ts = ts[:,:, -CPT:, :]       # 時系列の末尾からCPTだけ遡ったところから末尾まで
        if CPT==1:
            ts = ts.reshape(iters, batch_size, vector_length)
 
    print('展開数{:4d}, バッチサイズ{:4d}, 時系列長{:4d}, ベクトル長{:4d} に整形しました' \
          .format(iters, batch_size, time_size, vector_length))
    return xs, ts, iters

# -- 文章などの一続きの学習データをバッチ処理に適した形状に整形 ------------ 
#    先頭のインデクスで各展開のデータを取出す
def arrange_mini_batch_old2(data, batch_size=100, time_size=35, CPT=None, step=None):
    print('ミニバッチ用に長さ', len(data), 'のデータを加工します')
    d = [] 
    step = time_size if step is None else step
    for s in range(0, time_size, step):
        d += data[s:] ; print('開始位置', s, 'からデータを重複して切出します')
    data = d
    data_size = len(data)
    iters = (data_size - time_size) // (batch_size * time_size) # 展開数(整数)
    available_length = batch_size * time_size * iters           # 有効な長さ
    print('重複切出しによってデータを', data_size, 'に拡張し', \
          'そこから有効長', available_length, 'のデータを取得しました')                   

    data = data[:available_length+1]
    xs = np.array(data[:-1])
    ts = np.array(data[1: ])
    xs = xs.reshape(batch_size, iters, time_size).transpose(1, 0, 2) 
    ts = ts.reshape(batch_size, iters, time_size).transpose(1, 0, 2)
   
    if CPT is not None:
        ts = ts[:,:, -CPT:] # 末尾からCPTだけ遡ったところから末尾まで
    if CPT==1:
        ts = ts.reshape(iters, batch_size)
    print('展開数{:5d}, バッチサイズ{:5d}, 時系列長{:3d}に整形しました' \
          .format(iters, batch_size, time_size))
    return xs, ts, iters

def arrange_mini_batch_old(x, t, batch_size=100, time_size=35):
    data_size = len(x)
    iters = data_size // (batch_size * time_size)  # 展開数(整数)
    available_length = batch_size*time_size*iters  # 有効な長さ
    xs = x[0 : available_length]                   # 端数切捨て
    ts = t[0 : available_length]                   # 端数切捨て　
    xs = xs.reshape(batch_size, iters, time_size).transpose(1, 0, 2) 
    ts = ts.reshape(batch_size, iters, time_size).transpose(1, 0, 2)
    print('展開数{:5d}, バッチサイズ{:5d}, 時系列長{:3d}に整形しました' \
          .format(iters, batch_size, time_size))
    return xs, ts, iters

# -- 文章などの一続きの学習データを入力データと正解データとして切り出す ------------ 
def arrange_time_data(data, time_size, CPT=None, step=None, position=False):
    print('一つながりのデータから時系列長の入力データとそれに対する正解データを切り出します')
    if position:
        print('併せて、そのデータの位置のインデクスをデータと同じ形状で返します')
    if isinstance(data, (list, tuple)):
        dtype = type(data[0])
        data = np.array(data, dtype=dtype) # データの型は要素の型を継承
    d = []; idx = []
    # キャプチャ幅も切出し間隔も指定されない場合はいずれもtime_size    
    if CPT is None and step is None:
        CPT = step = time_size
    # キャプチャ幅が指定され、切出し間隔が指定されない場合はCPTに合わせる
    if CPT is not None and step is None: 
        step = CPT
    # キャプチャ幅は指定されないが、切り出し幅が指定された場合
    if CPT is None and step is not None:
        CPT = time_size

    print('時系列長は',time_size, 'データの切出し間隔は', step, 'です')
    print('正解データの時系列長は', CPT, 'です')        
    for i in range(0, len(data) - time_size, step):  # 時系列長＋１の長さのデータを一括して
        d.append(data[i : i + time_size + 1])        # step幅ずつずらして切出す
        idx.append(range(i , i + time_size + 1))
    d = np.array(d, dtype=data[0].dtype)
    idx = np.array(idx, dtype=int)
    #print('###debug arrange_time_data', d.shape, idx.shape)
    #print(d[:10]); print(idx[:10])
    input_data    = d[:, 0:time_size]                 # 0～time_size-1 が入力
    correct_data  = d[:, time_size-CPT+1:time_size+1] # 1～time_size(1時刻ずらし)が正解
    input_idx   = idx[:, 0:time_size]
    correct_idx = idx[:, time_size-CPT+1:time_size+1]

    if CPT == 1:
        correct_data = correct_data.reshape(-1)
    print('入力データと正解値の形状：', input_data.shape, correct_data.shape)

    if not position:
        return input_data, correct_data
    return input_data, correct_data, input_idx, correct_idx

# -- 文章などの一続きの学習データを入力データと正解データとして切り出す ------------ 
def arrange_time_data_bkup(data, time_size, CPT=None, step=None):
    print('一つながりのデータから時系列長の入力データとそれに対する正解データを切り出します')
    if isinstance(data, (list, tuple)):
        dtype = type(data[0])
        data = np.array(data, dtype=dtype) # データの型は要素の型を継承
    d = []
    # キャプチャ幅も切出し間隔も指定されない場合はいずれもtime_size    
    if CPT is None and step is None:
        CPT = step = time_size
    # キャプチャ幅が指定され、切出し間隔が指定されない場合はCPTに合わせる
    if CPT is not None and step is None: 
        step = CPT
    # キャプチャ幅は指定されないが、切り出し幅が指定された場合
    if CPT is None and step is not None:
        CPT = time_size

    print('時系列長は',time_size, 'データの切出し間隔は', step, 'です')
    print('正解データの時系列長は', CPT, 'です')        
    for i in range(0, len(data) - time_size, step):  # 時系列長＋１の長さのデータを一括して
        d.append(data[i : i + time_size + 1])        # step幅ずつずらして切出す
    d = np.array(d, dtype=data[0].dtype)
    input_data   = d[:, 0:time_size]                 # 0～time_size-1 が入力　
    correct_data = d[:, time_size-CPT+1:time_size+1] # 1～time_size(1時刻ずらし)が正解
    if CPT == 1:
        correct_data = correct_data.reshape(-1)
    print('入力データと正解値の形状：', input_data.shape, correct_data.shape)
    return input_data, correct_data

# 画像データを画像の縦方向=時刻、横方向=データ幅の時系列データとみなす
# 縦方向が指定されたシード長＋１のデータを切出して、それを入力と正解に分離
def arrange_time_data_from_image(data, seed_length, CPT=None):
    if CPT is None:
        CPT = seed_length
    N, H, W = data.shape     # N:データ枚数 H:縦（データ時間長） W:横（データ幅）
    print('形状 枚数＝', N, '縦＝', H, '横＝', W, 'のデータから切出します')
    n = H - seed_length        # １枚から切出すデータの数
    print('１枚のデータから', n, '枚のデータを切出します')

    d = np.empty((N * n, seed_length+1, H))
    for i in range(N):
        for j in range(n):
            d[i*n+j, :, :] = data[i, j:j+seed_length+1, :]

    # 入力と正解に分離        
    input_data   = d[:, :-1, :]
    correct_data = d[:, -1,  :] if CPT==1 else d[:, -CPT:, :]
    print('入力の形状は：', input_data.shape, '正解の形状は', correct_data.shape, 'です')

    return input_data, correct_data

class CutOutBatch:
    """
    dataから連続する部分をbatch_size個だけ切り出す
    バッチ内は長さを揃えるが、別バッチでは下記の指定に従いランダムな長さ
    block_size:連続する長さ
               整数で指定したら、長さは固定、切出し間隔は1
               タプルで指定したら、最短長さと最長長さと切出し間隔
    shuffle:切り出す部分をシャッフルするかどうか
    
    """
    def __init__(self, data, block_size=500, batch_size=16, shuffle=True):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        if type(block_size) in (tuple, list):
            if len(block_size)==1:
                cutout_length = block_size[0]
                self.block_size=block_size[0]
                self.step = 1
            elif len(block_size)==2:
                cutout_length = block_size[1]
                self.block_size=block_size
                self.step = 1
            elif len(block_size)==3:
                cutout_length = block_size[1]
                self.block_size=block_size[:2]
                self.step = block_size[2]
            else:
                raise Exception('block_size is not applicable')
        else:
            cutout_length = block_size
            self.block_size = block_size
            self.step = 1
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_ix = np.arange(0, len(data)-cutout_length+1, self.step)
        self.n_batch = len(self.start_ix) // batch_size
        self.reset()

    def reset(self):
        self.start_ix.sort()
        self.iters = 0
        self.epoch = 0

    def shuffle_ix(self):
        np.random.shuffle(self.start_ix)

    def __call__(self):
        if type(self.block_size) in (tuple, list):
            various_length = np.random.randint(self.block_size[0], self.block_size[-1]+1)
        else:
            various_length = self.block_size
        idx = self.start_ix[self.iters:self.iters+self.batch_size]
        y = np.stack([self.data[int(i):int(i)+various_length] for i in idx])
        self.iters += self.batch_size
        if self.iters >= len(self.start_ix):
            if self.shuffle:
                np.random.shuffle(self.start_ix)
            self.epoch += 1
            self.iters = 0
        return y    

def cut_out_batch(data, block_size=500, batch_size=16):
    func = CutOutBatch(data, block_size+1, batch_size)
    np.random.shuffle(func.start_ix)
    bd = func()
    return bd

class CutOutBatchIx:
    """
    CutOutBatchとおなじ働きで、インデクス取出しだけを行う
    連続する番号をbatch_size個だけ切り出す
    バッチ内は長さを揃えるが、別バッチでは下記の指定に従いランダムな長さ
    block_size:連続する長さ
               整数で指定したら、長さは固定、切出し間隔は1
               タプルで指定したら、最短長さと最長長さと切出し間隔
    shuffle:切り出す部分をシャッフルするかどうか
    
    """
    def __init__(self, length, block_size=500, batch_size=16, shuffle=True):
        if type(block_size) in (tuple, list):
            if len(block_size)==1:
                cutout_length   = block_size[0]
                self.block_size = block_size[0]
                self.step = 1
            elif len(block_size)==2:
                cutout_length   = block_size[1]
                self.block_size = block_size
                self.step = 1
            elif len(block_size)==3:
                cutout_length   = block_size[1]
                self.block_size = block_size[:2]
                self.step = block_size[2]
            else:
                raise Exception('block_size is not applicable')
        else:
            cutout_length = block_size
            self.block_size = block_size
            self.step = 1
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_ix = np.arange(0, length-cutout_length+1, self.step)
        self.n_batch = len(self.start_ix) // batch_size
        self.reset()

    def reset(self):
        self.start_ix.sort()
        self.iters = 0
        self.epoch = 0

    def shuffle_ix(self):
        np.random.shuffle(self.start_ix)

    def __call__(self):
        if type(self.block_size) in (tuple, list):
            various_length = np.random.randint(self.block_size[0], self.block_size[-1]+1)
        else:
            various_length = self.block_size
        idx = self.start_ix[self.iters:self.iters+self.batch_size]
        y = np.stack([np.arange(int(i),int(i)+various_length, 1) for i in idx])
        self.iters += self.batch_size
        if self.iters >= len(self.start_ix):
            if self.shuffle:
                np.random.shuffle(self.start_ix)
            self.epoch += 1
            self.iters = 0
        return y    

# -- 複数画像を表示 --
def display_images(image):
    n = len(image)
    if image.ndim == 2:
        image_size = int(np.sqrt(image.shape[1]))
        image = image.reshape(image.shape[0], image_size, image_size) 
    plt.figure(figsize=(n*2, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(image[i].tolist(), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)  # 軸は非表示
        ax.get_yaxis().set_visible(False)
    plt.show()

def eval_perplexity(model, corpus, B=10, T=35, n=10000):
    model.reset_state()
    xs = corpus[:-1]; ts = corpus[1:] 
    xs, ts, iters = arrange_mini_batch(xs, ts, B, T)
    loss = 0 
    for i in range(iters):
        x = xs[i]
        t = ts[i]
        y = model.forward(x)
        c = convert_one_hot(t, n)
        loss += model.loss_function(y, c)
    ppl = float(np.exp(loss / iters))
    return ppl

# targetを10000個ずつ最後まで変換
'''#
target_t = None; n = 0
while n < len(target):
    r = min(10000, len(target)-n) 
    target_p = convert_one_hot(target[n:n+r], vocab_size)
    if target_t is None:
        target_t = target_p
    else:    
        target_t = np.vstack([target_t, target_p])
    n += 10000  
target = target_t
#'''#

# -- seed から順に次時刻のデータを size になるまで生成(画像などの値) --
# embedding を伴わない RNN で画像など、値から値を生成する場合に適合
def generate_data_from_data(func, seed, size):
    T, m = seed.shape
    gen_data = np.empty((size, m))
    gen_data[0:T, :] = seed                      # 時刻 T までの値は seed
    for j in range(size - T):                   
        x = gen_data[j:j+T, :].reshape(1, T, -1) # seed の大きさのデータを
        y = func(x, CPT=1)                       # 順伝播して次時刻のデータを生成
        gen_data[j+T, :] = y.reshape(-1)         # 生成したベクトルを１行追加
    return gen_data

# -- seed から順に次時刻のデータを size になるまで生成(文字列などのid) --
# embedding を備えた RNN で文字列生成をする場合に適合
def generate_id_from_id(func, seed, length=200, beta=2, skip_ids=None):
    created_data = seed.copy() # 生成されるデータ(初期値はseed) .copy()必須
    for i in range(length):
        x = np.array(seed, dtype='int')
        #print('generate_id_from_id', i, x.shape, x)
        #x = x.reshape((1,)+x.shape) # 非バッチ処理なので形状をバッチ処理に合わせる
        x = x.reshape(1, -1)   # 非バッチ処理なので形状をバッチ処理に合わせる
        y = func(x, CPT=1)     # y の形状は(1, width)

        if y.ndim == 3:        # CPTが効かない場合があるため
            y = y[:,-1,:]      # 仮処置
        
        y = y.reshape(-1)      # 非バッチ処理だから出力はベクトル
        p = np.empty_like(y, dtype='f4') # 極小値多数でエラーしないため精度が必要
        p[...] = y ** beta     # 確率分布の調整
        p /= np.sum(p)         # pの合計を1に         
        next_one = np.random.choice(len(p), size=1, p=p)
        if (skip_ids is None) or (next_one not in skip_ids):
            next_one = int(next_one) 
            created_data.append(next_one)
            seed = seed[1:] + [next_one] 
    return created_data

def generate2(func, start_id, skip_ids=None, sample_size=100):
    # 本の機能維持のために一応置いておきますが使いたくありません
    word_ids = [start_id]
    x = start_id
    count = 0
    while count < sample_size:
        x = np.array(x).reshape(1, 1)
        p = func(x).reshape(-1)
        p = p / np.sum(p) # 合計が１にならない場合の対処療法
        sampled = np.random.choice(len(p), size=1, p=p)
        if (skip_ids is None) or (sampled not in skip_ids):
            x = sampled
            word_ids.append(int(x))
        count += 1    
    return word_ids

def generate_text(func, seed, width, length=200, beta=2):
    # embeddingを伴わないRNNで文字列生成
    created_data = seed        # 生成されるデータ(初期値はseed)
    for i in range(length):
        x = cf.convert_one_hot(seed, width) # 入力をone-hot表現に
        x = x.reshape((1,)+x.shape) # 非バッチ処理なので形状をバッチ処理に合わせる
        y = func(x, CPT=1)     # y の形状は(1, width) 
        y = y.reshape(-1)      # 非バッチ処理だから出力はベクトル
        p = y ** beta          # 確率分布の調整
        p = p / np.sum(p)      # pの合計を1に
        next_one = np.random.choice(len(p), size=1, p=p)
        next_one = int(next_one)
        created_data.append(next_one)
        seed = seed[1:] + [next_one]
    return created_data
    
def create_text(func, char_to_index, index_to_char, prev_text, length=200, beta=2):
    # embeddingを伴わないRNNで文字列生成
    n_time  = len(prev_text)
    n_chars = len(char_to_index)
    #lstm_layer.set_state()
    created_text = prev_text  # 生成されるテキスト
    print('Seed:', created_text)
    for i in range(length):  # 200文字の文章を生成
        # 入力をone-hot表現に
        x = np.zeros((1, n_time, n_chars))
        for j, char in enumerate(prev_text):
            x[0, j, char_to_index[char]] = 1
        
        # 予測を行い、次の文字を得る
        y = func(x, CPT=1)
        p = y[0] ** beta  # 確率分布の調整
        p = p / np.sum(p) # pの合計を1に
        next_index = np.random.choice(len(p), size=1, p=p)
        next_char = index_to_char[int(next_index[0])]
        created_text += next_char
        prev_text = prev_text[1:] + next_char

    print(created_text)
    print()  # 改行
    #lstm_layer.set_state()

# -- 潜在変数を平面にプロット --
def plot_latent_variables(func, x, t, axis=(0,1)):
    min_t, max_t = int(np.min(t)), int(np.max(t))
    plt.figure(figsize=(10, 10))
    n  = len(x)
    n2 = n if n < 2000 else 1000 
    for i in range(0, n, n2):
        xi = x[i:i+n2]
        ti = t[i:i+n2]
        zi = func(xi)
        for j in range(min_t, max_t+1):
            zt  = zi[ti==j]          # z から t==i の要素を抽出
            z_1 = zt[:, axis[0]]     # y軸
            z_2 = zt[:, axis[1]]     # x軸
            marker = '$'+str(j)+'$'  # 数値をマーカーに
            plt.scatter(z_2.tolist(), z_1.tolist(), marker=marker, s=75)
    plt.xlabel('z_2')
    plt.ylabel('z_1')
    #plt.xlim(-3, 3)
    #plt.ylim(-3, 3)
    plt.grid()
    plt.show()

# -- 潜在変数を与えて画像を描画 --
def picture_image(func, x, C, Ih, Iw, pil=False, save=False):
    image = func(x)
    if C <= 1:
        image = image.reshape(Ih, Iw)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) # 画像をはっきり
        plt.imshow(image.tolist(), cmap='Greys_r')
    else:
        image = image.reshape(C, Ih, Iw).transpose(1, 2, 0)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        plt.imshow(image.tolist())
    plt.show()
    if pil:
        #image = image.reshape(C, Ih, Iw).transpose(1, 2, 0)
        #image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.asnumpy(image*255).astype(np.uint8)
        image = Image.fromarray(image)
        #image.filter(ImageFilter.SMOOTH_MORE)
        image = image.resize((300, 300), Image.LANCZOS)
        image.show()
        if save:
            image.save("image.png")


# -- 画像を生成して表示 --
def generate_random_images(func, nz, C, Ih, Iw, n=81, reverse=False, rate=1.0, offset=None):
    if offset is None:
        offset = 0
    elif type(offset) in (float, int):
        pass
    elif isinstance(offset, np.ndarray):
        offset = offset.reshape(1, -1)
    else:
        raise Exception('offset is not applicable.')

    # 画像の生成
    n_rows = int(n ** 0.5)  # 行数
    n_cols = n // n_rows    # 列数
    # 入力となるノイズ
    #noise = np.random.normal(0, 1.0, (n_rows*n_cols, nz))  # 平均 0 標準偏差 1 の乱数
    noise = np.random.randn(n_rows*n_cols, nz) * rate + offset
    # 画像を生成して 0-1 の範囲に調整
    if C <= 1:
        #y = func(noise); print(y.shape, n, Ih, Iw)
        g_imgs = func(noise).reshape(n, Ih, Iw)
    else:
        g_imgs = func(noise).reshape(n, C, Ih, Iw).transpose(0, 2, 3, 1)
    g_imgs = (g_imgs - np.min(g_imgs)) / (np.max(g_imgs) - np.min(g_imgs))  
    g_imgs = 1 - g_imgs if reverse==True else g_imgs
    Ih_spaced = Ih + 2; Iw_spaced = Iw + 2
    if C <= 1:
        matrix_image = np.empty((Ih_spaced*n_rows, Iw_spaced*n_cols))  # 全体の画像
    else:
        matrix_image = np.empty((Ih_spaced*n_rows, Iw_spaced*n_cols, C)) # 全体の画像
    matrix_image[...] = 1.0 if reverse==True else 0.0 
    #print(matrix_image.shape, g_imgs.shape)
    #  生成された画像を並べて一枚の画像にする
    for i in range(n_rows):
        for j in range(n_cols):
            g_img = g_imgs[i*n_cols + j]
            top  = i*Ih_spaced
            left = j*Iw_spaced
            matrix_image[top : top+Ih, left : left+Iw] = g_img

    plt.figure(figsize=(9, 9))
    if C <=1:
        plt.imshow(matrix_image.tolist(), cmap='Greys_r')
    else:
        plt.imshow(matrix_image.tolist())#, cmap='Greys_r')
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  # 軸目盛りのラベルと線を消す
    plt.show()

def generate_random_images2(func, z, C, Ih, Iw, n=81, reverse=False, rate=1.0):
    """ 引数zと同様(平均と標準偏差が同様)の乱数を潜在変数としてfuncにより画像を生成 """
    mu = np.mean(z, axis=0, keepdims=True)
    sigma = np.std(z, axis=0, keepdims=True)
    nz = z.shape[-1]

    # 画像の生成
    n_rows = int(n ** 0.5)  # 行数
    n_cols = n // n_rows    # 列数
    # 入力となるノイズ
    #noise = np.random.normal(0, 1.0, (n_rows*n_cols, nz))  # 平均 0 標準偏差 1 の乱数
    noise = np.random.randn(n_rows*n_cols, nz) * sigma * rate + mu
    #print('mu_z =', mu, 'sigma_z =', sigma)
    #print('mu_n', np.mean(noise, axis=0), 'sigma_n =', np.std(noise, axis=0))
    
    # 画像を生成して 0-1 の範囲に調整
    if C <= 1:
        #y = func(noise); print(y.shape, n, Ih, Iw)
        g_imgs = func(noise).reshape(n, Ih, Iw)
    else:
        g_imgs = func(noise).reshape(n, C, Ih, Iw).transpose(0, 2, 3, 1)
    g_imgs = (g_imgs - np.min(g_imgs)) / (np.max(g_imgs) - np.min(g_imgs))  
    g_imgs = 1 - g_imgs if reverse==True else g_imgs
    Ih_spaced = Ih + 2; Iw_spaced = Iw + 2
    if C <= 1:
        matrix_image = np.empty((Ih_spaced*n_rows, Iw_spaced*n_cols))  # 全体の画像
    else:
        matrix_image = np.empty((Ih_spaced*n_rows, Iw_spaced*n_cols, C)) # 全体の画像
    matrix_image[...] = 1.0 if reverse==True else 0.0 
    #print(matrix_image.shape, g_imgs.shape)
    #  生成された画像を並べて一枚の画像にする
    for i in range(n_rows):
        for j in range(n_cols):
            g_img = g_imgs[i*n_cols + j]
            top  = i*Ih_spaced
            left = j*Iw_spaced
            matrix_image[top : top+Ih, left : left+Iw] = g_img

    plt.figure(figsize=(9, 9))
    if C <=1:
        plt.imshow(matrix_image.tolist(), cmap='Greys_r')
    else:
        plt.imshow(matrix_image.tolist())#, cmap='Greys_r')
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  # 軸目盛りのラベルと線を消す
    plt.show()

# -- 画像を生成して表示 --　
def picture_varying_latent_variables(func, z, C, Ih, Iw, axis=(0,1), n=81, rang=2.8, reverse=False):
    # 画像の設定
    n_rows = int(n ** 0.5)  # 行数
    n_cols = n // n_rows    # 列数
    Ih_spaced = Ih + 2; Iw_spaced = Iw + 2
    if np.all(z==0):
        nz = z.shape[1]
        x  = np.zeros(nz) 
        z_1 = np.linspace( rang, -rang, n_rows)  # 行
        z_2 = np.linspace(-rang,  rang, n_cols)  # 列
    else:
        x = np.mean(np.array(z), axis=0) # hdarrayとcupyの相性問題対処でわざわざnp.array()
        z_1 = np.linspace(np.max(z[:, axis[0]]), np.min(z[:, axis[0]]), n_rows)  # 行
        z_2 = np.linspace(np.min(z[:, axis[1]]), np.max(z[:, axis[1]]), n_cols)  # 列
    print('行', z_1, '\n列', z_2)
    if C <= 1:
        matrix_image = np.empty((Ih_spaced*n_rows, Iw_spaced*n_cols))  # 全体の画像
    else:
        matrix_image = np.empty((Ih_spaced*n_rows, Iw_spaced*n_cols, C)) # 全体の画像
    matrix_image[...] = 1.0 if reverse==True else 0.0 

    #  潜在変数を変化させて画像を生成
    for i, zi in enumerate(z_1):
        for j, zj in enumerate(z_2):
            x[axis[0]] = float(zi)
            x[axis[1]] = float(zj)
            if C <= 1:
                image = func(x).reshape(Ih, Iw)
            else:
                image = func(x).reshape(C, Ih, Iw).transpose(1, 2, 0)
            image = 1 - image if reverse==True else image
            image = (image - np.min(image)) / (np.max(image) - np.min(image)) # 画像をはっきり
            top  = i*Ih_spaced
            left = j*Iw_spaced
            matrix_image[top : top+Ih, left : left+Iw] = image

    plt.figure(figsize=(9, 9))
    if C <=1:
        plt.imshow(matrix_image.tolist(), cmap='Greys_r')
    else:
        plt.imshow(matrix_image.tolist())#, cmap='Greys_r')
    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)  # 軸目盛りのラベルと線を消す
    plt.show()

def is_np(x):
    """ xがnpと定義された形式かどうかを判定 """
    return type(x).__module__ == np.__name__

# -- 学習結果の継承 --
def inherit_from(file_name, naive_title, naive_params):
    print('この関数 inherit_from の使用は推奨しません')
    print('かわりに load_parameters の使用を推奨します')
    # -- ファイルの読み取り --
    fin = open(file_name, 'rb')
    data = fin.read()
    fin.close()

    # -- ヘッダーを取得 --
    title_head = struct.unpack('32s', data[0:32])[0].decode().strip()
    print('ヘッダー:', title_head)
    offset = 32

    # -- 重みとバイアスのデータを取得 --
    params = {}
    for item in naive_params.keys(): # 元のパラメタのキー
        # -- キーを読取る --
        key = struct.unpack('16s', data[offset:offset+16])[0].decode().strip()
        # -- パラメタの形状を取得 --
        ps1, ps2, ps3, ps4 = struct.unpack('4i', data[offset+16:offset+32])
        offset += 32
        if   ps2 == 0: # affine層または畳込み層のバイアス
            psize = ps1
            pdim  = 1
        elif ps3 == 0: # affine層の重み　
            psize = ps1 * ps2
            pdim  = 2
        else:          # 畳込み層の重み      
            psize = ps1 * ps2 * ps3 * ps4
            pdim  = 4
        # -- パラメタの値を取得して形状に合わせる --　　　　　　
        picked_param = struct.unpack(psize*'d', data[offset:offset+psize*8])
        picked_param = np.array(picked_param)
        picked_param = picked_param.reshape(ps1,ps2)         if pdim == 2 else picked_param
        picked_param = picked_param.reshape(ps1,ps2,ps3,ps4) if pdim == 4 else picked_param
        params[key]  = picked_param
        offset += psize*8
        print(key, params[key].shape, params[key])
   
    # パラメタの形状の一致を確認
    params_match = (title_head == naive_title)                         # タイトルの一致
    #print(title_head, naive_title, params_match) ### debug
    for key in params.keys():
        params_match *= (params[key].shape == naive_params[key].shape) # パラメタの形状の一致
        #print(key, params[key].shape, naive_params[key].shape, params_match)   ### debug

    return params_match, params

# -- 学習結果の保存 --
def inherit_to(file_name, title, params):
    print('この関数 inherit_to の使用は推奨しません')
    print('かわりに save_parameters の使用を推奨します')
    # -- データ編集変換 ---　　　
    data=bytes()
    # -- タイトルヘッダ --
    data += struct.pack('32s', title.ljust(32).encode()) 
    print('ヘッダー:', title)
    # -- データの内容を記録 --
    for key in params.keys(): # パラメタのキーを順に取出す　
        # -- キーをデータに入れる(16バイト) --
        data += struct.pack('16s', key.ljust(16).encode())
        # -- パラメタの大きさをデータに入れる(16バイト) --
        picked_param = params[key]  #                     affine層の場合  畳込み層の場合
        pdim = picked_param.ndim    # パラメタの次元数 -> w:2  b:1        w:4  b:1
        ps1 = picked_param.shape[0]                     # 入力数          フィルタ数  
        ps2 = picked_param.shape[1] if pdim >= 2 else 0 # ニューロン数    入力チャネル数
        ps3 = picked_param.shape[2] if pdim >= 3 else 0 # なし            フィルタ高
        ps4 = picked_param.shape[3] if pdim >= 4 else 0 # なし            フィルタ幅 
        print(key, picked_param.shape)
        data += struct.pack('4i', ps1, ps2, ps3, ps4)
        #print(key, ps1, ps2, ps3, ps4) ### debug 
        # -- パラメタの要素をデータに入れる --
        for item in picked_param.reshape(-1):
            data += struct.pack('d', item)
    #  -- ファイルに書き込み --　　
    print(len(data),'バイトのデータをファイルに記録')
    fout = open(file_name, 'wb')
    fout.write(data)
    fout.close()

