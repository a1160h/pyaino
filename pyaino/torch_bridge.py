# torch_bridge
# 20250818 井上

from pyaino.Config import *
from pyaino import common_function as cf
import re
import torch
import warnings


def state_dict_to_dict(state_dict):
    """ pytorchのstate_dictをmodelのパラメタの保存継承に使う通常の辞書に変換 """
    pattern = r'\.(\d+)\.' # ピリオドに囲まれた数字
    dictionary = {}        # 変換後の辞書
    for tk, tv in state_dict.items():
        #print(tk)
        # pytorch流をNeuron.Sequentialの名称に変換
        k = re.sub(pattern, lambda m: f'.layers[{m.group(1)}].', tk)
        k = k.replace('weight', 'w') # pytorchとの名称の違い
        k = k.replace('bias', 'b')   # 同上        
        v = tv.numpy()               # torch.tensorをndarrayに変換
        #print(k, ':', v.shape, type(v))
        dictionary[k] = v
    return dictionary    

def set_parameter(obj, name, pname, params):
    """ 条件をチェックし然るべき形にしてパラメタをセット """
    fullname = name + '.' + pname
    if fullname not in params.keys(): # paramsに無いものは放置
        warnings.warn(fullname+' skipped.')
        return

    param = params[fullname]
    cond1 = obj.__class__.__name__ in ('LinearLayer',)
    cond2 = obj.__class__.__base__.__name__ in ('BaseLayer',)
    cond3 = pname in ('w',)
    if (cond1 or cond2) and cond3:
        print('param.shape :', param.shape, end=' -> ')
        param = param.T
        print(param.shape)
    setattr(obj, pname, param)
    return


def import_parameters_recursive(obj, params=None, name=None, param_label = ('w', 'v', 'b', 'gamma', 'beta'),
                             seen=None, generation=0):
    """ 再帰的にparamsをobjのtarget項に設定 """
    if name is None:   # はじめは呼び出し元でのobj名
        name = cf.who_am_i(obj, stack_offset=3) # 0:who_am_i->1:ここ->2:呼び出し元->3:__main__
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
                if hasattr(obj, p):
                    set_parameter(obj, name, p, params)

        generation = generation + 1
        # クラスに付随する辞書を対象に再帰呼び出し
        name, seen, generation = import_parameters_recursive(obj.__dict__, params=params,
                                 name=name, seen=seen, generation=generation)

    return name, seen, generation

def import_parameters(model, params, **kwargs): # targetを指定するなどのため
    name, seen, generation = import_parameters_recursive(model, params, **kwargs)

def load_torch_params(file_name, model=None):
    """ torch.save(model.state_dict(), path)で格納されたパラメタをロードする """
    torch_params = torch.load(file_name,
                              map_location=torch.device('cpu'),
                              weights_only=True)

    params0 = state_dict_to_dict(torch_params)

    name = cf.who_am_i(model, stack_offset=2)
    print('top_level =', name)

    params = {}
    for k, v in params0.items():
        params[name+'.'+k] = v
    for k, v in params.items():
        print(k, ':', v.shape)   

    print('load_parameters called on np =', np.__name__)    
    if np.__name__=='cupy':
        params = cf.n2c_dictionary(params)
    if np.__name__=='numpy':
        params = cf.c2n_dictionary(params)
        
    import_parameters(model, params)
    print(model.__class__.__name__, 'モデルのパラメータをファイルから取得しました<=', file_name)
    return params
