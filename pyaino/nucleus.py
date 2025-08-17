# nucleus
# define by runによる自動微分の核心モジュール
# 20250817 A.Inoue

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
            raise Exception('Inconsistent HDArray for backtrace. May created by numpy arithmetic.')
        debug_print('*'*72 ,'\n<HDA backtrace>', id(self), '世代', self.generation, '変数形状', self.shape)
        # 勾配が指定されたら自身のアトリビュートに設定
        self.set_grad(grad)
        # -- 引数に従ってConfig.create_graphを設定して実行 --
        with using_config('create_graph', create_graph or Config.higher_derivative): # 20241030
            seen_var, seen_func = backtrace_graph(self, seen_var, seen_func) # seen_var, seen_funcは更新
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
        self.grad = HDArray(grad) if Config.create_graph and not isinstance(grad, HDArray) \
                                  else grad
        debug_print('grad', id(self.grad), 'is set for', id(self))
        
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
    HDArrayは演算子オーバーロード対象なので、対象から外すには通常のndarrayに戻さなければならない．
    しかしそのためには、np.array()で別オブジェクトを起こす必要があり、頻繁な処理では好ましくない．
    そこで、np.asarray()でデータを引き継ぐことができ、かつ、演算子オーバーロード非対象の
    オブジェクトとしてXArrayを設ける．    

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

class Function:
    """ 微分可能関数の基底クラス """
    def __init__(self, log=False, log_file='log_file.txt', preserve_attr=False):
        self.inputs = None
        self.outputs = None
        self.generation = 0
        if not hasattr(self, 'config'): # 派生クラスのコントラクタの設定を壊さないように
            self.config = None          # layerとして使う場合に必要
        self.outputs_copy = None # weakrefでdeadしないようにするため
        self.y_shapes = None     # 同上仮処置20240927
        self.graph_exist = False # 仮20241003
        self.preserve_attr = preserve_attr # 出力の上書き時のアトリビュート保護

        if Config.log_function:
            Config.function_list.append(id(self))
        
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
        debug_print('<forward ↓>', self.__class__.__name__, id(self), 'forward (',
                    Config.create_graph, Config.higher_derivative, Config.derivative,
                    Config.backtrace_duration, Config.operator_state, ')')

        # -- 派生クラスの引数がタプルやリストに複数オペランドをパックした形式の場合20250503AI          
        if len(inputs)==1 and all(isinstance(x, (tuple, list)) for x in inputs):
            inputs, = inputs
            warnings.warn(self.__class__.__name__+' inputs packed in list or tuple.')

        #            [x.shape if isinstance(x, np.ndarray) else x for x in inputs], end=' ')       # -- 無限ループ防止(forward中の演算でFunctionのforwardが呼び出されると無限ループ) --
        if Config.operator_state == 3: # 演算子オーバーロードでの無限ループ防止
            xs = [XArray(x) if isinstance(x, HDArray) else x for x in inputs]
        else:
            xs = inputs

        # -- 派生クラスの順伝播 --
        with using_config('create_graph', False): # forward中のforwardはグラフ生成しない
            ys = self.__forward__(*xs, **kwargs)  # 演算に使うxsはinputsと別物で構わない
        if self.preserve_attr: # 関数の返り値をアトリビュートself.outputsと同じ値の別物に分ける
            outputs = (ys.copy(),) if type(ys) is not tuple else ys.copy()
        else:
            outputs = (ys,) if type(ys) is not tuple else ys
        self.y_shapes = [y.shape if isinstance(y, np.ndarray) else () for y in outputs] # 仮処置20241023   

        # -- グラフ非生成時の短縮パス --
        if not Config.create_graph:# and Config.operator_state != 3: # 仮処置20240924
            self.inputs  = inputs
            self.outputs = outputs
            return ys                                  # ysを外から書き換えてもself.outputsに影響しない　　　　　　　　　　
        
        # -- 入出力対象にグラフ生成する --
        debug_print('<fw>', self.__class__.__name__, 'creating graph')
        #if issubclass(self.__class__, HDFunction):
        for x in inputs:
            self.check_decency(x) # チェックだけ 仮に全チェック20250503AI
        # inputsをHDArrayにするが、元々そうでない場合には別物になる(高階微分やグラフ可視化では問題)
        self.inputs = [x if isinstance(x, HDArray) and hasattr(x, 'generation')
                       else HDArray(x) for x in inputs] # backwardで勾配セットの準備

        self.generation = max([x.generation for x in self.inputs])
        # 出力はどのみち新しく作られるものだから、別物になるのは構わない
        outputs = [HDArray(y) for y in outputs] # ysがHDAを継承したとしても必要
        outputs = [self.set_creator_and_generation(y) for y in outputs]

        debug_print('<fw> 保存された出力の世代と生成者',
                    [o.generation for o in outputs],
                    [[oc.__class__.__name__ for oc in o.creator] for o in outputs])
        # self.outputsはweakrefだが、その中身はoutputsと同一で生成者や世代も引継ぐ
        self.outputs = [weakref.ref(y) if y is not None else y for y in outputs]

        if Config.preserve_weakref_obj:
            self.outputs_copy = [y() for y in self.outputs] # weakrefでdeadしない為に
        #outputs = [y() for y in self.outputs]
        debug_print('', self.__class__.__name__, '世代', self.generation,
            '\n 入力', '世代', [x.generation for x in self.inputs],  len(self.inputs),
                [id(x) for x in self.inputs], [x.shape if isinstance(x, np.ndarray) else x for x in self.inputs],
            '\n 出力', '世代', [y.generation for y in outputs], len(outputs),
                [id(y) for y in outputs],[y.shape if isinstance(y, np.ndarray) else y for y in outputs],
            '\n<forward ↑>\n')
        self.graph_exist = True # 仮20241003
        return outputs[0] if len(outputs)<=1 else outputs
    
    def backward(self, *gys, seen_var=None, **kwargs):
        # -- 出力の勾配を得るなどの準備 --
        # backward中のforwardでグラフ生成の場合には、その実行前に、
        # forwardのinputsになる変数をHDArrayにしておく必要がある
        # すなわちgysは予めHDArrayにしておく必要がある
        if len(gys)==0: # backtraceの際に勾配は引数で与えられない
            #gys = [self.get_grad(y) for y in self.outputs]
            gys = self.get_grads() # 仮処置20240927
        else: # 勾配が引数で与えられるがgy=1などの場合にも対処
            gys = self.fix_grads(gys)

        if seen_var is None:
            seen_var = set()
            #print('<bw>initialize seen_var', seen_var)
        debug_print(self.__class__.__name__, id(self), 'backward', 'gys =', [id(gy) for gy in gys],
                    Config.create_graph, Config.higher_derivative, Config.derivative,
                    Config.backtrace_duration, Config.operator_state)

        # -- 派生クラスの逆伝播(グラフ生成はcreate_graphの指定に従う) --
        # HDArrayのbacktrace()メソッド実行時にデフォルトFalseで指定し、
        # Config.create_graphに反映するので、それに従うべき
        # すなわち、下記のwith using_configは要らない
        #with using_config('create_graph', Config.higher_derivative):
        gxs = self.__backward__(*gys, **kwargs)
        if gxs is None: # 20250605AI 
            return

        # -- グラフ非生成時の短縮パス --
        #if not (self.graph_exist and Config.derivative):
        # バックトレース期間中でないならば__backward__()メソッドの結果をそのまま返せば良い20250506AI
        if not Config.backtrace_duration:
            return gxs
        
        #if not (self.graph_exist and Config.derivative):
        if not self.graph_exist: # forward中のforwardでグラフ生成しないことに対応
            #warnings.warn(self.__class__.__name__+ ' backward graph not exist nor derivative')
            return gxs

        # -- 勾配を入力変数に設定 --
        debug_print('<bw> For inputs of', self.__class__.__name__, 'set gradient', \
                    '\n   ',  self.inputs, '\n   ', gxs)
        #gxs = (gxs,) if type(gxs) is not tuple else gxs
        gxs = (gxs,) if not isinstance(gxs , (tuple, list)) else gxs

        for x, gx in zip(self.inputs, gxs):
            debug_print('<bw0> for x', type(x), id(x))

            if not isinstance(x, HDArray):
                warnings.warn(self.__class__.__name__+'non HDArray variable for backward.')
                x = self.fix_inconsistent_variable(x, seen_var) # xは別物になるため返り値で反映必要20250506AI                　
            if id(x) in seen_var:
                debug_print('<bw1>', self.__class__.__name__, 'input is in seen_var', id(x))

                x.grad += gx  # x.gradのidを変えない(この操作で関数の定義次第ではgysが影響を受けるので要注意)

                if Config.create_graph:#Config.higher_derivative:# + Config.create_graph: # 仮処置20241001
                    debug_print('<bw1.1>高階微分可能でx.gradの調整', type(x.grad), type(gx))
                    self.check_decency(x.grad)
                    self.check_decency(gx)
                    x.grad.generation = max(x.grad.generation, gx.generation)
                    x.grad.creator.update(gx.creator)
                    self.gx_creator_update(x, gx) # x.gradに併合されたgx側の計算グラフの辻褄合わせ

                    debug_print('<bw1.5>x.gradの世代と生成者', id(x.grad), id(gx),
                            x.grad.generation, [gxc.__class__.__name__ for gxc in x.grad.creator])

            else:
                debug_print('<bw1>', self.__class__.__name__, 'input is new', id(x))
                x.grad = gx   # gxのidをx.gradが引継ぐ              
                seen_var.add(id(x))

            debug_print('<bw2>',  self.__class__.__name__, 'x.grad', id(x.grad), 'gx', id(gx))    

            if gx is not None: # 仮処置20241002 勾配が帰らないような引数を持つ関数もありうる
                if x.grad is None:
                    print(id(x), 'in', seen_var, 'whereas', x.grad is None)
                    raise Exception("x is in seen_var, but, who's gradient is None")
               
        return gxs[0] if len(gxs)<=1 else gxs
        
    def check_decency(self, x, warning=False):
        """ グラフの作れるようなまともな変数であることの確認(まともなHDArray、但し、定数項は除く) """
        if isinstance(x, HDArray) and hasattr(x, 'generation'):
            return 0
        elif isinstance(x, (int, float)):
            debug_print('### check_decency 0', self.__class__.__name__, x, type(x))
            return 1
        elif warning:    
            msg = 'Excuting ' + self.__class__.__name__ + ', '
            msg += 'variable is not ready to create graph.'
            #msg += '\n' + str(id(x)) + str(type(x))    
            msg += '\n' + str(type(x))    
            msg += '\nmay need to specify create_graph=True for backtrace()'
            #msg += '\n' + str(x)
            x = HDArray(x)
            warnings.warn(msg)
            """
            if type(x) in (int, float):
                debug_print('### check_decency 1', self.__class__.__name__, x, type(x))
                return 1
            elif x.ndim==0:
                debug_print('### check_decency 2', self.__class__.__name__, x, type(x))
                return 1
            else:
            """
            debug_print('### check_decency 3', self.__class__.__name__, x, type(x))
            return 2
        else:
            debug_print('### check_decency 3', self.__class__.__name__, x, type(x))
            return 2

    def gx_creator_update(self, x, gx):
        """ gxの生成者の出力==gxそのものをx.gradで置換える(弱参照に注意) """
        debug_print('gxとその生成者の出力(入替前)', id(gx), [[id(y()) for y in gxc.outputs] for gxc in gx.creator])
        for gxc in gx.creator:
            gxc.outputs = [weakref.ref(x.grad) for y in gxc.outputs if id(gx)==id(y())]
        debug_print('gxとその生成者の出力(入替後)', id(gx), [[id(y()) for y in gxc.outputs] for gxc in gx.creator])

    def fix_inconsistent_variable(self, x, seen_var):
        id_x_old = id(x)          # 元のid
        msg = 'During backward() of {} got non HDArray variable. {} id {} '\
        .format(self.__class__.__name__, type(x), id_x_old)   
        x = HDArray(x)            # 新たにHDArrayにするとidが変わる
        x.grad = np.zeros_like(x, dtype=Config.dtype) # 勾配を初期化
        id_x_new = id(x)          # HDArrayにした後のid
        msg += str(id_x_new)      # これをmsgに追加
        warnings.warn(msg)        # waringの出力、停止しないで続行
        seen_var.discard(id_x_old) # 削除ただし、無くてもエラーしない
        seen_var.add(id_x_new)     # 追加 
        return x

    def get_outputs(self):
        """ アトリビュートに保存した出力を順伝播の際の出力と同じ形式で得る """
        outputs = [y() if isinstance(y, weakref.ReferenceType)
                       else y for y in self.outputs]
        return outputs[0] if len(outputs)<=1 else outputs        

    def get_grads(self, default=1.0): # 仮処置20240927
        """ 勾配が設定されていればそれを、さもなくばdefault値を返す """
        gys = []
        for y in self.outputs:
            if isinstance(y, weakref.ReferenceType): # weakrefの判別
                y = y()
            if y is None:
                msg = self.__class__.__name__+' getting grads of'+str(self.outputs)
                warnings.warn(msg)
            if hasattr(y, 'grad') and y.grad is not None:
                gy = y.grad
            elif isinstance(y, np.ndarray):
                gy = np.broadcast_to(np.array(default, dtype=Config.dtype), y.shape)
            else: 
                gy = default # 仮処置20250712
            gys.append(gy)     
        return gys
    

    def get_grad(self, y, default=1.0):
        """ 出力に勾配が設定されていればそれを、さもなくばdefault値を返す """
        # yがHDArrayであってもなくても有効
        if isinstance(y, weakref.ReferenceType): # weakrefの判別
            y = y()
        if hasattr(y, 'grad') and y.grad is not None:
            return y.grad
        return np.broadcast_to(np.array(default, dtype=Config.dtype), y.shape)

    def fix_grads(self, gys):
        """ 与えられた勾配の型と形状を出力に合わせる """
        if len(self.y_shapes)!=len(gys): # y_shapesもgysも常にタプル
            raise Exception("Can't fix grad's shape as output's shape.")
        #gys = [np.broadcast_to(gy if isinstance(gy, np.ndarray) else np.array(gy) ,
        #                       y_shape) for gy, y_shape in zip(gys, self.y_shapes)]
        gys = [np.broadcast_to(asndarray(gy), y_shape) for gy, y_shape in zip(gys, self.y_shapes)]
        return gys

    def set_creator_and_generation(self, y):
        """ HDAの親関数を設定して、親関数+1に世代を設定する """
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
    pass


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
    ⑥ 得られた勾配を入力変数に設定、このとき既出の変数かどうかで勾配を加算または新たに設定
    ⑦ Functionの入力の親関数を次に備えてfuncsに加える
       その後④に戻って繰り返す
    '''
    if not y.creator:  # 試し20240301
        print('Quit backtracing.')
        return
    debug_print('Start backtracing', [yc.__class__.__name__ for yc in y.creator],
                Config.create_graph, Config.higher_derivative, Config.derivative, Config.operator_state)
    if seen_var  is None:
        seen_var  = set() # 既出の変数のidを記録する、seen_varの初期化により勾配は初期化 
    if seen_func is None:
        seen_func = set() # 既出の関数のidを記録する 

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
        debug_print('\n<backtrace_graph↓>', f.__class__.__name__, id(f), 
                                '世代', f.generation, 'while in', len(flist)+1)
        f.backward(seen_var=seen_var) # 関数の逆伝播を呼出す
        for x in f.inputs:
            funcs.update(x.creator)
            seen_func.update((id(f) for f in x.creator))

        debug_print('<backtrace_graph↑>　残り=>', len(funcs), '個')
    Config.backtrace_duration = False
    return seen_var, seen_func    

def gradient(y, x, create_graph=True):
    """ y=f(x)に対しdydxを返す """
    y.backtrace(create_graph=create_graph)
    return x.grad


class CompositFunction:
    """
    合成関数、_forwardメソッドに順方向だけ記述した合成関数の順逆両方の伝播メソッドを得る

    これにより静的計算グラフによる順・逆の両方向の伝播のものと組み合わせることが出来る
    逆伝播のメソッドは、define-by-runの機能を使って作るため、_forwardメソッドの定義は、
    Functionsに定義したnucleusのFunctionクラスを親とするクラスによらなければならない．
    引数については、Functionクラス同様に外でHDA化していればそのまま、否なら中でHDA化して、
    self.inputsに保存するとともに_forwardメソッドの引数とする．
    従って、_forwardメソッドの中で定義に使われるFunctionクラスでは、属性も含めてそのまま
    各々のself.inputsとなり、逆伝播の際にはそれを通じて、このCompositFunctionクラスの
    self.inputsに属性が反映されることになる．　　　
    """
    
    def __init__(self):
        self.inputs = None
        self.outputs = None
   
    def forward(self, *inputs):
        debug_print(self.__class__.__name__)
        self.inputs = [i if isinstance(i, HDArray) and hasattr(i, 'generation')
                       else HDArray(i) for i in inputs]
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
        debug_print(self.__class__.__name__, self.inputs)
        ginputs = [np.zeros_like(i, dtype=Config.dtype) if i.grad is None
                   else np.array(i.grad) for i in self.inputs] # 勾配はndarrayにする(HDArrayではない)
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
        debug_print('operator overload init', 'HDF =', HDF, Config.operator_state, end=' ')
        self.operator_state = Config.operator_state # 元の状態を保持

        if Config.operator_state == 0:
            self.save()
            Config.operator_state += 1

        elif Config.operator_state == 2:
            raise Exception('Operator state is wrong.')

        debug_print('->', Config.operator_state)        

    def __enter__(self):
        debug_print('operator overload enter', Config.operator_state, end=' ')

        if self.state:
            if Config.operator_state == 1:
                self.overload()
                Config.operator_state += 2

        else:
            if Config.operator_state == 3:
                self.recover()
                Config.operator_state -= 2

        debug_print('->', Config.operator_state)        

    def __exit__(self, exception_type, exception_value, traceback):
        debug_print('operator overload exit', Config.operator_state, end=' ')

        if self.state:
            if self.operator_state == 1:
                self.recover()
                Config.operator_state -= 2
        else:
            if self.operator_state == 3:
                self.overload()
                Config.operator_state += 2

        debug_print('->', Config.operator_state)        

       
    def __call__(self):
        debug_print('operator overload enter', Config.operator_state, end=' ')

        if self.state:
            if Config.operator_state == 1:
                self.overload()
                Config.operator_state += 2

        else:
            if Config.operator_state == 3:
                self.recover()
                Config.operator_state -= 2

        debug_print('->', Config.operator_state)        

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
抑止の手段は、with文により一時的にOperatorOverloadを無効にするか、あるいは、
オペランドをXArrayに変換してOperatorOverloadの対象外にするかのいずれかだが、
後者の方が簡単

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
    set_higher_derivative(True)   # これでメモリリーク発生
    #clear_log()
   
    # メモリリークのテスト
    class Square(Function):
        def __forward__(self, x):
            y = np.square(x)
            return y

        def __backward__(self, gy):
            x, = self.inputs
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
    
