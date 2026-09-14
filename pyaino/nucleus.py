# nucleus
# define by runによる自動微分の核心モジュール
# 20260914 A.Inoue

from pyaino.Config import *
import weakref
import os, sys
import warnings, copy

def asndarray(x):
    """ int, float, numpy.ndarray, cupy.ndarrayなどをConfig.np.ndarrayにする """
    if isinstance(x, np.ndarray):
        return x
    elif type(x) in (float, int, bool, list):
        return np.array(x)
    else: # numpy/cupyが違うとnp.ndarrayには見えない
        try:
            return np.array(x) # numpy.ndarray -> cupy.ndarray
        except:
            try:
                return x.get() # cupy.ndarray -> numpy.ndarray
            except:
                try:
                    return np.array(x.tolist()) # ダメ元
                except TypeError as e:
                    print(e)

class HDArray(np.ndarray): 
    def __new__(cls, data, dtype=None):
        """
        Higher Derivative Array (HDA) 高階微分可能配列
        np.ndarrayを継承した新たなクラスを定義し、ndarrayに無い属性を付加する　　
        asarrayでdataをコピーせずそのままndarrayにし、
        .viewで元のクラスと同じメモリを参照するobjを作る

        """
        #obj = np.asarray(data, dtype=Config.dtype)
        obj = np.asarray(data, dtype=dtype) # 20241030型指定をやめる(元の型を継承) 
        obj = obj.view(cls)
        obj.generation = 0
        obj.creator = set() 
        obj.grad = None
        obj.name = None
        return obj

    def backtrace(self, grad=None, create_graph=False, seen_var=None, seen_func=None):
        """ HDAを入口にしてグラフを辿って逆伝播開始する """
        # グラフを辿る過程でcreate_graph=Trueでない限りグラフを生成しない
        if not hasattr(self, 'generation'):
            raise Exception(
                'Inconsistent HDArray for backtrace. May created by numpy arithmetic.')
        debug_print('\n<backtrace>',
                    f"creator={[c.__class__.__name__ for c in self.creator]}",
                    f'gen={self.generation}', f'shape={self.shape}',
                    f'create_graph={create_graph or Config.higher_derivative}')
        # 勾配が指定されたら自身のアトリビュートに設定
        self.set_grad(grad)
        # -- 引数に従ってConfig.create_graphを設定して実行 seen_var, seen_funcは更新 --
        with using_config('create_graph', create_graph or Config.higher_derivative): 
            seen_var, seen_func = backtrace_graph(self, seen_var, seen_func)  
        return seen_var, seen_func    

    def reset(self):
        self.generation = 0
        self.creator = set() 
        self.grad = None
        self.name = None
        
    def set_grad(self, grad=None, default=1.0):
        """ 自身のアトリビュートに勾配を設定 """
        if isinstance(grad, np.ndarray) and grad.shape==self.shape:
            pass
        else:
            if grad is None:
                grad = default
            elif grad is not None and type(grad) in (int, float):
                pass
            else:
                raise Exception('Invalid gradient specified.' + str(grad))
            grad = np.broadcast_to(np.array(grad, dtype=Config.dtype), self.shape)
        # バックトレース中にグラフ生成するには勾配もHDArrayでなければならない
        self.grad = HDArray(grad) \
            if Config.create_graph and not isinstance(grad, HDArray)  else grad
        
    @property
    def copyz(self): # 仮処置(numpy/cupyと干渉するので除外)
        """ 属性を継承しながら別のオブジェクト """
        new = np.array(self, dtype=Config.dtype)
        new = HDArray(new)
        new.generation = self.generation
        new.creator = self.creator
        new.grad = self.grad
        new.name = self.name
        return new

class XArray(np.ndarray):
    """
    HDArrayは演算子オーバーロード対象なので、
    対象から外すには通常のndarrayに戻さなければならない．
    しかしそのためには、np.array()で別オブジェクトを起こす必要があり、
    頻繁な処理では好ましくない．
    そこで、np.asarray()でデータを引き継ぐことができ、
    かつ、演算子オーバーロード非対象のオブジェクトとしてXArrayを設ける．    

    """
    def __new__(cls, data):
        #obj = np.asarray(data, dtype=Config.dtype)
        obj = np.asarray(data) # 2024225型指定をやめる(元の型を継承) 
        obj = obj.view(cls)
        obj.name = data.name if hasattr(data, 'name') else None
        return obj

def record_log():
    """ Config.log_setをlog_fileに記録する """
    main_dir = os.getcwd()
    log_file = Config.log_file
    file_path = os.path.join(main_dir, log_file)
    debug_print(file_path, 'に Config.log_set が記録されます。')
    with open(file_path, "a") as f:
        for l in Config.log_set:
            f.write(str(l)+'\n')
    Config.log_set.clear()    

def clear_log():
    """ Config.function_listを空にする """
    Config.function_list.clear()

    #main_dir = os.getcwd()
    #log_file = Config.log_file
    #file_path = os.path.join(main_dir, log_file)
    #with open(file_path, "w") as f:
    #    pass # 何も書かない

def as_tuple(x):
    """ タプル形式に正規化する """
    if isinstance(x, tuple):
        y = x
    elif isinstance(x, list):
        y = tuple(x)
    else:
        y = (x,)
    return y    

class Function:
    """ 微分可能関数の基底クラス """
    
    def __init__(self, log=False, log_file='log_file.txt', preserve_attr=False):
        self.inputs = None
        self.outputs = None
        self.generation = 0
        if not hasattr(self, 'config'): # 派生クラスのコントラクタの設定を壊さないように
            self.config = None          # layerとして使う場合に必要
        self.outputs_copy = None # weakrefでdeadしないようにするため
        self.output_shapes = None     # 同上仮処置20240927
        self.graph_exist = False # 仮20241003
        self.preserve_attr = preserve_attr # 出力の上書き時のアトリビュート保護
        self.called_in_forward  = None 
        self.called_in_backward = None 
        self.io_aliases = () # forward内で保持した入出力の別名属性名

        if Config.log_function:
            Config.function_list.append(id(self))

        # 後日preserve_attrは削除予定20260912AI
        if preserve_attr:
            raise ValueError(
                "preserve_attr=True is no longer supported. "
                "Backward state must be preserved by the Function subclass."
            )            


    def call_forward(self, *xs, **kwargs):
        """ サブクラスの__forward__を数値計算として実行する """
        # 入力はHDArrayのまま渡し、XArrayには変換しない。
        # forward中のforwardはcreate_graph=Falseとして外側Functionへ統合する。
        # 演算子オーバーロードが有効なら、その時点で有効なfamilyを一時的に抑止する。
        
        # 入出力のalias管理のためサブクラス実行前状態を把握
        attrs_before = self.__dict__.copy() if self.called_in_forward else None

        # サブクラスの実行
        if Config.operator_state == 3:
            with (using_config('create_graph', False),
                  using_config('in_forward', True),
                  OperatorOverload(False, HDF=Config.higher_derivative)):
                  # overloadは現在の状態に合わせる   
                ys = self.__forward__(*xs, **kwargs)
        else:
            with (using_config('create_graph', False),
                  using_config('in_forward', True)):
                ys = self.__forward__(*xs, **kwargs)

        # サブクラス実行中に作られた入出力のaliasを記録
        if attrs_before is not None:
            self.record_io_aliases(attrs_before, xs, ys)
        else:
            self.io_aliases = ()

        return ys

    def call_backward(self, *gys, **kwargs):
        """ Functionの__backward__を数値計算として実行する。 """
        if Config.operator_state == 3:
            with (using_config('in_backward', True),
                  OperatorOverload(False, HDF=Config.higher_derivative)):
                gxs = self.__backward__(*gys, **kwargs)
        else:        
            with using_config('in_backward', True):
                gxs = self.__backward__(*gys, **kwargs)
        return gxs 

    def record_io_aliases(self, attrs_before, xs, ys):
        """ forward内で新たに保持した入出力配列の別名属性名を記録する """
        ys = as_tuple(ys)
        refs = tuple(v for v in (*xs, *ys) if isinstance(v, np.ndarray))
        sentinel = object()
        aliases = []

        for name, value in self.__dict__.items():
            if not isinstance(value, np.ndarray):
                continue

            old_value = attrs_before.get(name, sentinel)
            if old_value is value:
                continue

            if any(value is ref for ref in refs):
                aliases.append(name)

        self.io_aliases = tuple(aliases)

    def release_io_aliases(self):
        """ backward完了後、記録された入出力の別名参照を解放する """
        for name in self.io_aliases:
            setattr(self, name, None)
        self.io_aliases = ()

    def replace_output_aliases(self, raw_outputs, outputs, inputs):
        """ HDFのforward内で保持したraw outputのaliasをgraph outputへ置換する """
        input_ids = {id(x) for x in inputs}
        for y, o in zip(raw_outputs, outputs):
            # 入出力が同一objectの場合は、入力stateまで出力に置換しない
            if id(y) in input_ids:
                continue

            for name, value in tuple(self.__dict__.items()):
                if value is y:
                    setattr(self, name, o)

    def forward(self, *inputs, **kwargs):# kwargsはグラフ生成対象外
        """ 逆伝播のためにグラフを作りつつ順伝播 """
        '''
        入力は、呼び出される際に引数がHDAならば、属性を含めてそのままそれを、
        そうでないなら引数をHDAにしてself.inputsに保存する．
        前者の場合には、引数の変数は、値も属性もself.inputsと同じとなり、
        後者の場合には、self.inputのみにgenerationなどの属性が付随する
        出力は、派生クラスに定義された演算の結果を、self.outputsに保存する．
        HDAの変数はnumpy演算を行うと見かけはHDAでも属性は引き継がれない．
        この見掛け倒しのHDAは要注意（属性の存在でチェックできる）．
        そこで値を引き継ぎながら改めてHDAに明示的に変換し、
        その上でcreatorとgenerationを付与する．
        引数のうち、*inputsは*xsとして派生クラスの__forward__()メソッドに渡され、
        **kwargsはそのまま渡される．前者は計算グラフ生成の対象であり、後者は対象外．
        '''
        # 入出力は前回の状態を明示的に破棄し、CuPy poolで再利用可能になる時期を前倒し
        self.inputs  = None 
        self.outputs = None 
        self.graph_exist = False
        self.outputs_copy = None
        self.called_in_forward  = Config.in_forward  # 私は誰かのforward の中で呼ばれた
        self.called_in_backward = Config.in_backward # 私は誰かのbackwardの中で呼ばれた

        debug_print(f'<forward> {self.__class__.__name__}',
                    f'called_in_forward={self.called_in_forward}',
                    f'called_in_backward={self.called_in_backward}')

        # -- 派生クラスの引数がタプルやリストに複数オペランドをパックした形式の場合          
        if len(inputs)==1 and all(isinstance(x, (tuple, list)) for x in inputs):
            inputs, = inputs
            warnings.warn(self.__class__.__name__+' inputs packed in list or tuple.')

        # -- 派生クラスの順伝播 inputsをそのままxsとして実行部へ渡す --
        ys = self.call_forward(*inputs, **kwargs)    
        ys = as_tuple(ys) 

        # 属性保護が必要な場合だけ各出力をコピー 後日削除予定20260912AI
        if self.preserve_attr:
            ys = tuple(y.copy() for y in ys)

        self.output_shapes = [y.shape if isinstance(y, np.ndarray) else () for y in ys]    

        # -- グラフ非生成時の短縮パス ysを外から書き換えてもself.outputsに影響しない --
        if not Config.create_graph:
            return ys[0] if len(ys)<=1 else ys # 後日単純化予定　
        
        # -- 入出力対象にグラフ生成する --
        for x in inputs:
            self.check_decency(x) # チェックだけ 仮に全チェック20250503AI
        # inputsをHDArrayにするが、元々そうでない場合には別物になる
        # (高階微分やグラフ可視化では問題)
        # backwardで勾配セットの準備  Noneの対処20260414AI
        self.inputs = \
            [x if x is None or (isinstance(x, HDArray) and hasattr(x, 'generation'))
               else HDArray(x) for x in inputs]
        if self.inputs: # inputsが無い場合には'0'のまま20260603AI              
            self.generation = max(x.generation for x in self.inputs if x is not None)
            
        # 出力をgraph-readyなHDArrayにする
        # HDFで__forward__内にraw outputのaliasを保持している場合は、
        # graph outputへ差し替えてbackwardから同じ出力graphを参照できるようにする
        outputs = [self.set_creator_and_generation(y) for y in ys]
        if isinstance(self, HDFunction):
            self.replace_output_aliases(ys, outputs, inputs)

        debug_print('  graph',
            f"gen={[x.generation for x in self.inputs if x is not None]}",
            f"-> {[o.generation for o in outputs]}",
            f"shape={[x.shape if isinstance(x, np.ndarray) else x for x in self.inputs]}",
            f"-> {[y.shape if isinstance(y, np.ndarray) else y for y in outputs]}")
        # self.outputsはweakrefだが、その中身はoutputsと同一で生成者や世代も引継ぐ
        self.outputs = [weakref.ref(y) if y is not None else y for y in outputs]

        if Config.preserve_weakref_obj:
            self.outputs_copy = [y() for y in self.outputs] # weakrefでdeadしない為に
        #outputs = [y() for y in self.outputs]
        self.graph_exist = True # 仮20241003
        return outputs[0] if len(outputs)<=1 else outputs
    
    def backward(self, *gys, seen_var=None, **kwargs):
        # -- 出力の勾配を得るなどの準備 --
        # backward中のforwardでグラフ生成の場合には、その実行前に、
        # forwardのinputsになる変数をHDArrayにしておく必要がある
        # すなわちgysは予めHDArrayにしておく必要がある
        #
        # HDArrayのbacktrace()メソッド実行時には、
        # 高階微分をConfig.higher_derivativeであるか、
        # または引数create_graphで直接指定するかによって、
        # Config.create_graphをwith using_configで操作して実行

        if len(gys) != 0:               # 勾配が与えられた場合 
            gys = self.fix_grads(gys)
        elif Config.backtrace_duration: # 勾配は変数から取得
            gys = self.get_grads()
        else:                           # デフォルト1
            gys = self.get_default_grads()

        debug_print(f'<backward> {self.__class__.__name__}',
                    f'create_graph={Config.create_graph}')

        if seen_var is None:
            seen_var = set()

        # -- 派生クラスの逆伝播 --
        gxs = self.call_backward(*gys, **kwargs)

        # forwardの中から呼ばれた場合の早期解放
        if self.called_in_forward and not isinstance(self, HDFunction): 
            self.release_io_aliases()
            self.outputs = None
        if gxs is None: # 20250605AI 
            return

        # -- グラフ非生成時の短縮パス --
        # バックトレース期間中でないならば__backward__()メソッドの結果をそのまま返せば良い
        if not Config.backtrace_duration:
            return gxs
        
        if not self.graph_exist: # forward中のforwardでグラフ生成しないことに対応
            return gxs

        # -- 勾配を入力変数に設定 --
        gxs = as_tuple(gxs)
        self.set_input_grads(gxs, seen_var)
               
        return gxs[0] if len(gxs)<=1 else gxs
        
    def set_input_grads(self, gxs, seen_var):
        """  backwardで得た勾配を入力変数へ設定する """
        for x, gx in zip(self.inputs, gxs):

            if not isinstance(x, HDArray):
                warnings.warn(self.__class__.__name__+'non HDArray variable for backward.')
                x = self.fix_inconsistent_variable(x, seen_var)
                        # xは別物になるため返り値で反映必要                　

            if id(x) in seen_var:
                x.grad += gx  # x.gradのidを変えない
                              # (この操作で関数の定義次第ではgysが影響を受けるので要注意)

                if Config.create_graph:
                    self.check_decency(x.grad)
                    self.check_decency(gx)
                    x.grad.generation = max(x.grad.generation, gx.generation)
                    x.grad.creator.update(gx.creator)
                    self.gx_creator_update(x, gx)
                              # x.gradに併合されたgx側の計算グラフの辻褄合わせ

            else:
                x.grad = gx
                seen_var.add(id(x))

            if gx is not None: # 勾配が帰らないような引数を持つ関数もありうる
                if x.grad is None:
                    print(id(x), 'in', seen_var, 'whereas', x.grad is None)
                    raise Exception("x is in seen_var, but, who's gradient is None")

    def check_decency(self, x, warning=False):
        """ グラフの作れるようなまともな変数であることの確認
            (まともなHDArray、但し、定数項は除く) """
        if isinstance(x, HDArray) and hasattr(x, 'generation'):
            return 0
        elif isinstance(x, (int, float)):
            return 1
        elif warning:    
            msg = 'Excuting ' + self.__class__.__name__ + ', '
            msg += 'variable is not ready to create graph.'
            msg += '\n' + str(type(x))    
            msg += '\nmay need to specify create_graph=True for backtrace()'
            x = HDArray(x)
            warnings.warn(msg)
            return 2
        else:
            return 2

    def gx_creator_update(self, x, gx):
        """ gxの生成者の出力==gxそのものをx.gradで置換える(弱参照に注意) """
        for gxc in gx.creator:
            gxc.outputs = [weakref.ref(x.grad) for y in gxc.outputs if id(gx)==id(y())]

    def fix_inconsistent_variable(self, x, seen_var):
        id_x_old = id(x)           # 元のid
        msg = 'During backward() of {} got non HDArray variable. {} id {} '\
        .format(self.__class__.__name__, type(x), id_x_old)   
        x = HDArray(x)             # 新たにHDArrayにするとidが変わる
        x.grad = np.zeros_like(x, dtype=Config.dtype) # 勾配を初期化
        id_x_new = id(x)           # HDArrayにした後のid
        msg += str(id_x_new)       # これをmsgに追加
        warnings.warn(msg)         # waringの出力、停止しないで続行
        seen_var.discard(id_x_old) # 削除ただし、無くてもエラーしない
        seen_var.add(id_x_new)     # 追加 
        return x

    def get_outputs(self):
        """ アトリビュートに保存した出力を順伝播の際の出力と同じ形式で得る """
        outputs = [y() if isinstance(y, weakref.ReferenceType)
                       else y for y in self.outputs]
        return outputs[0] if len(outputs)<=1 else outputs        

    def get_grads(self):
        """ 変数に設定された勾配（Noneなら0）を取得する """
        gys = []
        for y, y_shape in zip(self.outputs, self.output_shapes):
            if isinstance(y, weakref.ReferenceType):
                y = y()
            if y is not None and hasattr(y, 'grad') and y.grad is not None:
                gy = y.grad
            else:
                gy = np.zeros(y_shape, dtype=Config.dtype)
            gys.append(gy)
        return gys

    def get_default_grads(self, default=1):
        """ default勾配をoutput_shapesに合わせて返す """
        gys = [
            np.broadcast_to(np.array(default, dtype=Config.dtype), y_shape)
            for y_shape in self.output_shapes
            ]
        return gys

    def fix_grads(self, gys):
        """ 与えられた勾配の型と形状を出力に合わせる """
        if len(self.output_shapes)!=len(gys): # output_shapesもgysも常にタプル
            raise Exception("Can't fix grad's shape as output's shape.")
        gys = [gy if isinstance(gy, np.ndarray) and gy.shape == y_shape \
               else np.broadcast_to(asndarray(gy), y_shape) 
               for gy, y_shape in zip(gys, self.output_shapes)]
        return gys

    def set_creator_and_generation(self, y):
        """ HDAの親関数を設定して、親関数+1に世代を設定する """
        y = HDArray(y)
        y.generation = self.generation + 1
        y.creator.add(self) 
        return y

    def __forward__(self, *args, **kwargs):
        raise NotImplementedError()
    
    def __backward__(self, *args, **kwargs):
        warnings.warn('Backward is not explicitly defined. Return with zeros.')
        gxs = tuple(HDArray(np.zeros_like(x)) for x in self.inputs)
        return gxs[0] if len(gxs)==1 else gxs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def generate_dot_graph(self, var_set=None, verbose=False):
        """ 入出力を含めて自身の計算グラフを生成 """
        if var_set is None:
            var_set = set() # 変数の集合
        dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
        name = self.__class__.__name__

        if name=='Pow' and hasattr(self, 'c'):
            name += str(self.c)

        if name=='Exp' or 'Log'and hasattr(self, 'log_of_base'):
            name += '(base '+str(np.exp(self.log_of_base))[:8]+')'
        
        if verbose:
            name += ': ' + str(id(f))
        ret = dot_func.format(id(self), name)

        # -- 入出力 --
        for x in self.inputs:
            if id(x) not in var_set:
                ret += self._dot_var(x, verbose)
            var_set.add(id(x))
        for y in self.outputs:
            if isinstance(y, weakref.ReferenceType):
                y = y() 
            if id(y) not in var_set:
                ret += self._dot_var(y, verbose)
            var_set.add(id(y))

        # -- edge --
        dot_edge = '{} -> {}\n'
        for x in self.inputs:
            ret += dot_edge.format(id(x), id(self))
        for y in self.outputs:
            if isinstance(y, weakref.ReferenceType):
                y = y() 
            ret += dot_edge.format(id(self), id(y)) # yの弱参照に注意！
           
        self.ret = ret
        self.var_set = var_set
        return ret, var_set

    def _dot_var(self, v, verbose=False):
        dot_var = '{} [label="{}", color=orange, style=filled]\n'
        name = ''
        if hasattr(v, 'name'):
            if v.name is not None:
                name = v.name
            elif v.ndim==0:
                name = str(v)
            if v.ndim>=1:     
                name += str(v.shape)
               
        elif isinstance(v, (int, float)):
            name = str(v)

        if verbose:
            if name != '':
                name += ': '
            name += str(id(v))
             
        return dot_var.format(id(v), name)


class HDFunction(Function):
    """ 高階微分用Function。forward/backwardの実行文脈だけFunctionと分ける """
    '''
    サブクラスの中では以下のようになっている必要がある
    backward に入力が必要なら、graph-ready な入力を self.x 等として保持する
    backward に出力値が必要なら、__forward__ 内で self.y 等として保持してよい
    graph生成時にはnucleusがraw outputのaliasをgraph outputへ差し替える

    '''
    def call_forward(self, *xs, **kwargs):
        """
        HDFunctionの__forward__を実行

        通常の演算子式だけを一時的に抑止する一方、create_graphは抑止しないため、
        明示的に呼ばれたHDFunctionは内部graphを生成
        """
        attrs_before = self.__dict__.copy() if self.called_in_forward else None

        if Config.operator_state == 3:
            with (using_config('in_forward', True),
                  OperatorOverload(False, HDF=Config.higher_derivative)):
                    ys = self.__forward__(*xs, **kwargs)
        else:
            with using_config('in_forward', True):
                ys = self.__forward__(*xs, **kwargs)

        if attrs_before is not None:
            self.record_io_aliases(attrs_before, xs, ys)
        else:
            self.io_aliases = ()

        return ys
    
    def call_backward(self, *gys, **kwargs):
        """ HDFunctionの__backward__はoperator overload状態をそのまま受け継ぐ """
        with using_config('in_backward', True):
            gxs = self.__backward__(*gys, **kwargs)
        return gxs    
   
   
def print_data_class_etc(xs, comment=None):
    xs = (xs,) if type(xs) not in(tuple, list) else xs # 常にタプルかリストにする
    for x in xs:
        if not isinstance(x, HDArray):
            print(comment, '>>>>変数がHDArrayでない', type(x))
            break
        if x.creator: # 集合が空でない
            x_creator_generation = max([xc.generation for xc in x.creator])
        else:
            x_creator_generation = ''
        print(comment, id(x), x.__class__.__name__, x, 
              '世代', x.generation, end=' ')
        print('親関数', end=' ')
        for c in x.creator:
            print(c.__class__.__name__, end=' ')
        print(x_creator_generation)


def backtrace_graph(y, seen_var=None, seen_func=None):
    '''
    HDAを入口として計算グラフを上流へ辿る
    その際に保存した勾配を更新したい場合には、seen_varにてその対象であることを受渡す
    以下、その手順
    ① 入口のHDAのbackwadメソッドからbacktrace_graph関数が呼出される
    ② 該HDAの勾配の初期値を設定する
    　 １つに限定されるのでリストにはしない(HDFのbackwardメソッドでは複数に対応)
    ③ 入口のHDAの親関数をfuncsに入れてバックトレース開始
    ④ funcsをリストにして世代順にソートして、HDFを一つ取出す
    ⑤ 継承元の関数のbackwardメソッドを呼出す
    ⑥ 得られた勾配を入力変数に設定、このとき既出の変数かどうかで
       勾配を加算または新たに設定
    ⑦ Functionの入力の親関数を次に備えてfuncsに加える
       その後④に戻って繰り返す
    '''
    if not y.creator:  
        print('Quit backtracing.')
        return
    if seen_var  is None:
        seen_var  = set() # 既出の変数のidを記録、seen_varの初期化により勾配は初期化 
    if seen_func is None:
        seen_func = set() # 既出の関数のidを記録 

    funcs = set()
    funcs.update(y.creator)
    seen_func.update((id(f) for f in y.creator))
    if Config.backtrace_duration: # 20250506AI　
        raise Exception('During backtrace another backtrace is called!')
    Config.backtrace_duration = True
    while funcs: # ループの中で関数fを関数リストfuncsから取出して上流へ辿っていく
        flist = list(funcs)
        flist.sort(key=lambda f: f.generation) # 世代の小さい順にソート
        f = flist.pop() # flistから抽出,末尾 = 世代の大きいものから取出す　
        funcs.remove(f) # funcsからも同じものを削除
        debug_print(f'<trace> {f.__class__.__name__}', f'gen={f.generation}')
        f.backward(seen_var=seen_var) # 関数の逆伝播を呼出す
        for x in f.inputs:
            if x is None:
                continue  # 20260616AI
            funcs.update(x.creator)
            seen_func.update((id(f) for f in x.creator))

    Config.backtrace_duration = False
    return seen_var, seen_func    

def gradient(y, x, create_graph=True):
    """ y=f(x)に対しdydxを返す """
    y.backtrace(create_graph=create_graph)
    return x.grad


class CompositFunction:
    """ 合成関数
    _forwardメソッドに順方向だけ記述した合成関数の順逆両方の伝播メソッドを得る

    これにより静的計算グラフによる順・逆の両方向の伝播のものと組み合わせることが出来る
    逆伝播のメソッドは、define-by-runの機能を使って作るため、_forwardメソッドの定義は、
    Functionsに定義したnucleusのFunctionクラスを親とするクラスによらなければならない．
    引数については、Functionクラス同様に外でHDA化していればそのまま、
    否なら中でHDA化して、self.inputsに保存するとともに_forwardメソッドの引数とする．
    従って、_forwardメソッドの中で定義に使われるFunctionクラスでは、
    属性も含めてそのまま各々のself.inputsとなり、逆伝播の際にはそれを通じて、
    このCompositFunctionクラスのself.inputsに属性が反映されることになる．　　　
    """
    
    def __init__(self):
        self.inputs = None
        self.outputs = None
        print(f'derivative={Config.derivative}',
              f'higher_derivative={Config.higher_derivative}',
              f'create_graph{Config.create_graph}')
   
    def forward(self, *inputs):
        debug_print(f'<composite forward> {self.__class__.__name__}')
        self.inputs = [i if isinstance(i, HDArray) and hasattr(i, 'generation')
                       else HDArray(i) for i in inputs]

        if Config.create_graph: # 自動微分有効な場合は通常関数として実行
            return self._forward(*inputs) 

        with using_config('create_graph', True):
            with OperatorOverload(True):
                ys = self._forward(*self.inputs)
        self.outputs = (ys,) if type(ys) is not tuple else ys # 中は常にタプル
        outputs = [np.array(o) for o in self.outputs]         # 外は常にndarray
        return outputs[0] if len(self.outputs) <= 1 else outputs

    def _forward(self, *inputs):
        """ 個別の合成関数の順伝播の定義をHDFを用いて行う """
        raise NotImplimentedError()

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def backward(self, *goutputs):
        """ 変数のバックトレースで求めるので、合成関数の内容によらず共通 """

        if Config.create_graph:
            raise NotImplementedError(f'backward() method of {self.__class__.__name__}')
        
        if len(goutputs)==0:
            goutputs = [None for o in self.outputs]
        elif len(goutputs)!=len(self.outputs):
            raise Exception('Invalid arguments specified.') 
        
        seen_var = set()
        for o, g in zip(self.outputs, goutputs):
            #o.set_grad(g) # 出力に勾配を設定 <= backtraceの引数に勾配を指定できれば不要
            #with using_config('derivative', True):
            #    seen_var = o.backtrace(grad=g, seen_var=seen_var)
            seen_var = o.backtrace(grad=g, seen_var=seen_var) # 仮20250515AI
        debug_print(f'<composite backward> {self.__class__.__name__}',
                    f'input_shape={[i.shape for i in self.inputs]}')
        ginputs = [np.zeros_like(i, dtype=Config.dtype) if i.grad is None
                   else np.array(i.grad) for i in self.inputs]
                       # 勾配はndarrayにする(HDArrayではない)
        # 上記は引数が関数に含まれない場合(合成関数ではありうる)も正しい 
        return ginputs[0] if len(ginputs) <= 1 else ginputs


class OperatorOverload:
    def __init__(self, state=True, HDF=False): # 仮処置20241001
        if HDF:
            from pyaino.HDFunctions import OverloadContents
            warnings.warn('pyaino.HDFunctions will overload operators.')
        else:
            from pyaino.Functions import OverloadContents
            warnings.warn('pyaino.Functions will overload operators.')
        self.state = state
        self.save = OverloadContents().save
        self.overload = OverloadContents().overload
        self.recover = OverloadContents().recover
        self.operator_state = Config.operator_state # 元の状態を保持

        if Config.operator_state == 0:
            self.save()
            Config.operator_state += 1

        elif Config.operator_state == 2:
            raise Exception('Operator state is wrong.')

        debug_print('<operator init>', f'HDF={HDF}',
                    f'state={self.operator_state}->{Config.operator_state}')        

    def __enter__(self):

        if self.state:
            if Config.operator_state == 1:
                self.overload()
                Config.operator_state += 2

        else:
            if Config.operator_state == 3:
                self.recover()
                Config.operator_state -= 2

        debug_print('<operator enter>', f'state={Config.operator_state}')        

    def __exit__(self, exception_type, exception_value, traceback):

        if self.state:
            if self.operator_state == 1:
                self.recover()
                Config.operator_state -= 2
        else:
            if self.operator_state == 3:
                self.overload()
                Config.operator_state += 2

        debug_print('<operator exit>', f'state={Config.operator_state}')        

       
    def __call__(self):

        if self.state:
            if Config.operator_state == 1:
                self.overload()
                Config.operator_state += 2

        else:
            if Config.operator_state == 3:
                self.recover()
                Config.operator_state -= 2

        debug_print('<operator call>', f'state={Config.operator_state}')        

def operator_overload():
    OperatorOverload()()





"""
OperatorOverloadは以下の3つの場合がある

⓪　通常はOperatorOverloadは立てない

①　__main__の中で演算を定義するのに演算子が使いたくてOperatorOverloadを立てる場合

②　高階微分のためにbackwardメソッドでOeratorOverloadを立てて、
　　その中の演算をHDFとしてグラフ生成しながら実行する場合

これらを考慮してHDFの実行で

__forward__ メソッドでは、
⓪ではそのまま
①②ではOperatorOverloadの抑止が必須
抑止の手段は、with文により一時的にOperatorOverloadを無効にすることによる

__backward__ メソッドでは、
⓪ではそのまま
①ではそのままでも抑止してもどちらでも可
②ではOperatorOverloadを生かして全てHDFで実行する必要がある
しかし、もともとのOperatorOverloadの状態を鑑みれば、そのままそれに従えば良いだけ

またCompositFunctionは
_forwardメソッドの実行の際にcreate_graph並びにOperatorOverloadを一時的に強制して
HDFで実行し、そして、その外部とのやり取りに際してndarrayに戻す
だから外からは気にする必要がない



"""

if __name__=='__main__':
    import matplotlib.pyplot as plt
    set_higher_derivative(True)    
   
    # メモリリークのテスト
    class Square(Function):
        def __forward__(self, x):
            self.x = x
            y = np.square(x)
            return y

        def __backward__(self, gy):
            x = self.x
            gx = gy * 2 * x 
            return gx

    for i in range(30):
        x = HDArray(np.random.randn(1000000))
        y = Square()(Square()(Square()(Square()(x))))
        y.backtrace()
        gx = x.grad
        print(i, y.shape, gx.shape)

    # 基本的な高階微分のテスト
    x = np.hdarray(np.linspace(-2, 2))

    f1 = lambda x : x + 1
    f2 = lambda x : x**4 + 2*x**3 + 3*x**2 + 4*x + 5
    f3 = lambda x : 1/(x + 3.5)
    f4 = lambda x : 2 ** x

    funcs = f1, f2, f3, f4

    rank = 5
    for f in funcs:
        y = f(x)
        label  = "y"; labels = ["y=f(x)"]; logs = [y]
        for i in range(rank):
            print('rank', i, 'backtrace')
            y.backtrace(create_graph=True)
            if not hasattr(x, 'grad'): # 勾配がセットされなかったら例外
                Exception('no grad for x held')
            label += "'"; labels.append(label); logs.append(x.grad)
            y = x.grad                 # 次のrankに備える

        for i, y in enumerate(logs):
            plt.plot(x.tolist(), y.tolist(), label=labels[i])
        plt.legend()#loc='lower right')
        plt.show()


    # 勾配の与え方のバリエーションのテスト
    class Func(Function):
        def __forward__(self, x):
            y = x + 1
            return y

        def __backward__(self, gy):
            return gy
        
    f = Func()

    x = np.arange(24).reshape(4, 6)
    y = f(x)

    print(x)
    print(y)

    gy = np.ones_like(y)

    gx = f.backward(gy) # 外から勾配を与える　
    print(type(gx))
    print(gx)

    gx = f.backward(1)  # 勾配をデフォルトとして1を指定
    print(type(gx))
    print(gx)

    gx = f.backward()   # 勾配を外から与えない
    print(type(gx))
    print(gx)
    

    from pyaino import HDFunctions 
    Config.enable_debug_print=True
    f = HDFunctions.Ones()
    x = np.hdarray(np.arange(24).reshape(4, 6))
    print('x\n', x)
    y = f(x)
    print('y\n', y)
    y.backtrace()
    print("y'\n", x.grad)
    y = x.grad
    y.backtrace()
    print("y''\n", x.grad)    
