import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import os
import sys

# pyainoモジュールを正しくインポートできるように、検索パスの先頭にスクリプトのディレクトリを追加します。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pyainoの設定モジュールをロードし、バックエンドをNumPy（CPU）に固定します。
from pyaino.Config import Config, set_np
set_np('numpy')

from pyaino import Neuron
from pyaino import Activators
from pyaino import Functions

def restore_activation(op_type, node):
    if op_type == 'Relu':
        return Activators.ReLU()
    elif op_type == 'LeakyRelu':
        alpha = 0.01
        for attr in node.attribute:
            if attr.name == 'alpha':
                alpha = attr.f
        return Activators.LReLU(c=alpha)
    elif op_type == 'Sigmoid':
        return Activators.Sigmoid()
    elif op_type == 'Tanh':
        return Activators.Tanh()
    elif op_type == 'Softmax':
        return Activators.Softmax()
    elif op_type == 'Identity':
        return Activators.Identity()
    elif op_type == 'Elu':
        alpha = 1.0
        for attr in node.attribute:
            if attr.name == 'alpha':
                alpha = attr.f
        return Activators.ELU(c=alpha)
    elif op_type == 'Softplus':
        return Activators.Softplus()
    elif op_type == 'Gelu':
        approx = 'none'
        for attr in node.attribute:
            if attr.name == 'approximate':
                approx = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s
        if approx == 'tanh':
            return Activators.GELUap()
        else:
            return Activators.GELU()
    elif op_type == 'Swish':
        alpha = 1.0
        for attr in node.attribute:
            if attr.name == 'alpha':
                alpha = attr.f
        return Activators.Swish(beta=alpha)
    elif op_type == 'Mish':
        return Activators.Mish()
    return None

def import_onnx_to_pyaino(onnx_path):
    """
    ONNXモデルファイルを読み込み、pyainoのSequentialモデルとして再構築（インポート）します。
    PyTorchには依存せず、公式のonnx Pythonライブラリのみを使用します。
    
    引数:
        onnx_path: 読み込むONNXファイルのパス
    戻り値:
        pyainoのNeuron.Sequentialモデルオブジェクト
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # ONNXの初期化子（定数として保存されている重みやバイアス）を
    # 素早く検索できるように、{テンソル名: NumPy配列} の辞書形式にデコードして保持します。
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    
    pyaino_layers = []       # 再構築したpyainoレイヤーオブジェクトの格納用リスト
    
    skip_nodes = set()
    
    for i, node in enumerate(graph.node):
        if node.name in skip_nodes:
            continue
            
        op_type = node.op_type
        
        # --- 全結合層 (Gemm) ノードの復元 ---
        if op_type == 'Gemm':
            input_name = node.input[0]
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            
            w_onnx = initializers[weight_name]  # ONNXでの形状: (out_features, in_features)
            b_onnx = initializers[bias_name] if bias_name else None
            
            out_features, in_features = w_onnx.shape
            bias_enabled = bias_name is not None
            
            layer = Neuron.NeuronLayer(in_features, out_features, bias=bias_enabled, full_connection=True)
            layer.config = (in_features, out_features)
            layer.parameters.init_parameter()
            
            layer.parameters.w = w_onnx.T
            if bias_enabled:
                layer.parameters.b = b_onnx
                
            pyaino_layers.append(layer)
            
        # --- 2次元畳み込み層 (Conv) ノードの復元 ---
        # --- 畳み込み層 (Conv) ノードの復元 (1D/2D両対応) ---
        elif op_type == 'Conv':
            input_name = node.input[0]
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            
            w_onnx = initializers[weight_name]  # ONNXでの形状: (M, C, Fh, Fw) または (M, C, Fw)
            b_onnx = initializers[bias_name] if bias_name else None
            
            strides = []
            pads = []
            kernel_shape = []
            for attr in node.attribute:
                if attr.name == 'strides':
                    strides = list(attr.ints)
                elif attr.name == 'pads':
                    pads = list(attr.ints)
                elif attr.name == 'kernel_shape':
                    kernel_shape = list(attr.ints)
                    
            bias_enabled = bias_name is not None
            
            if len(kernel_shape) == 1:
                M, C, Fw = w_onnx.shape
                stride = strides[0] if strides else 1
                pad = pads[0] if pads else 0
                layer = Neuron.Conv1dLayer(M, Fw, stride=stride, pad=pad, bias=bias_enabled)
                layer.config = (C, None, M, Fw, stride, pad, None)
                layer.parameters.init_parameter()
                w_pyaino = w_onnx.reshape(M, -1).T
                layer.parameters.w = w_pyaino
                if bias_enabled:
                    layer.parameters.b = b_onnx
                pyaino_layers.append(layer)
            else:
                M, C, Fh, Fw = w_onnx.shape
                stride = strides[0] if strides else 1
                pad = pads[0] if pads else 0
                layer = Neuron.Conv2dLayer(M, Fh, stride, pad, bias=bias_enabled)
                layer.config = (C, None, None, M, Fh, Fw, stride, stride, pad, None, None)
                layer.parameters.init_parameter()
                w_pyaino = w_onnx.reshape(M, -1).T
                layer.parameters.w = w_pyaino
                if bias_enabled:
                    layer.parameters.b = b_onnx
                pyaino_layers.append(layer)
            
        # --- 活性化関数の復元 ---
        elif op_type in ('Relu', 'LeakyRelu', 'Sigmoid', 'Tanh', 'Softmax', 'Identity', 'Elu', 'Softplus', 'Gelu', 'Swish', 'Mish'):
            # 埋め込まれた活性化関数の復元
            if "embedded" in node.name:
                if len(pyaino_layers) > 0:
                    prev_layer = pyaino_layers[-1]
                    if hasattr(prev_layer, 'postphase'):
                        prev_layer.postphase.activator = restore_activation(op_type, node)
            else:
                # 独立した活性化関数レイヤー
                pyaino_layers.append(restore_activation(op_type, node))
                
        # --- Flatten の復元 ---
        elif op_type == 'Flatten':
            pyaino_layers.append(Neuron.Flatten())
            
        # --- MaxPool / AveragePool の復元 (1D/2D両対応) ---
        elif op_type in ('MaxPool', 'AveragePool'):
            kernel_shape = [2, 2]
            pads = [0, 0, 0, 0]
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    kernel_shape = list(attr.ints)
                elif attr.name == 'pads':
                    pads = list(attr.ints)
            pad = pads[0] if len(pads) > 0 else 0
            pool_size = kernel_shape[0]
            method = 'max' if op_type == 'MaxPool' else 'avg'
            
            if len(kernel_shape) == 1:
                pyaino_layers.append(Neuron.Pooling1dLayer(pool_size, pad, method))
            else:
                pyaino_layers.append(Neuron.Pooling2dLayer(pool_size, pad, method))

        # --- Dropout / Dropout2 の復元 ---
        elif op_type == 'Dropout':
            ratio_val = 0.5
            if len(node.input) > 1:
                ratio_name = node.input[1]
                if ratio_name in initializers:
                    ratio_val = float(initializers[ratio_name])
            
            if "Dropout2" in node.name:
                pyaino_layers.append(Neuron.Dropout2(preset=ratio_val))
            else:
                pyaino_layers.append(Neuron.Dropout(preset=ratio_val))

        # --- GlobalAveragePool の復元 ---
        elif op_type == 'GlobalAveragePool':
            pyaino_layers.append(Neuron.GlobalAveragePooling())

        # --- BatchNormalization の復元 ---
        elif op_type == 'BatchNormalization':
            parts = node.name.split('__')
            orig_class_name = parts[0]
            sb_enabled = False
            for k in range(len(parts) - 1):
                if parts[k] == 'sb':
                    sb_enabled = (parts[k+1] == '1')
                    break
                    
            scale_name = node.input[1]
            bias_name = node.input[2]
            mean_name = node.input[3]
            var_name = node.input[4]
            
            scale_val = initializers[scale_name]
            bias_val = initializers[bias_name]
            mean_val = initializers[mean_name]
            var_val = initializers[var_name]
            
            C = len(scale_val)
            
            eps_val = 1e-12
            for attr in node.attribute:
                if attr.name == 'epsilon':
                    eps_val = attr.f
                    
            if orig_class_name in ('BatchNorm2d', 'batch_norm_2d'):
                layer = Neuron.BatchNorm2d(scale_and_bias=sb_enabled, eps=eps_val)
                param_shape = (1, C, 1, 1)
            elif orig_class_name in ('BatchNorm1d', 'batch_norm_1d'):
                layer = Neuron.BatchNorm1d(scale_and_bias=sb_enabled, eps=eps_val)
                param_shape = (1, C, 1)
            else:
                layer = Neuron.BatchNormalization(scale_and_bias=sb_enabled, eps=eps_val)
                param_shape = (1, C)
                
            dummy_shape = [1] * len(param_shape)
            dummy_shape[1] = C
            layer.init_parameters(dummy_shape)
            
            layer.gamma = scale_val.reshape(param_shape)
            layer.beta = bias_val.reshape(param_shape)
            layer.mu_ppl = mean_val.reshape(param_shape)
            layer.sigma_ppl = np.sqrt(var_val).reshape(param_shape)
            
            pyaino_layers.append(layer)

        # --- LayerNormalization の復元 ---
        elif op_type == 'LayerNormalization':
            parts = node.name.split('__')
            orig_class_name = parts[0]
            sb_enabled = False
            for k in range(len(parts) - 1):
                if parts[k] == 'sb':
                    sb_enabled = (parts[k+1] == '1')
                    break
                    
            scale_name = node.input[1]
            bias_name = node.input[2]
            
            scale_val = initializers[scale_name]
            bias_val = initializers[bias_name]
            
            eps_val = 1e-12
            for attr in node.attribute:
                if attr.name == 'epsilon':
                    eps_val = attr.f
                    
            if orig_class_name in ('LayerNorm2d', 'layer_norm_2d'):
                layer = Neuron.LayerNorm2d(scale_and_bias=sb_enabled, eps=eps_val)
            elif orig_class_name in ('LayerNorm1d', 'layer_norm_1d'):
                layer = Neuron.LayerNorm1d(scale_and_bias=sb_enabled, eps=eps_val)
            else:
                node_axis = -1
                for attr in node.attribute:
                    if attr.name == 'axis':
                        node_axis = attr.i
                layer = Neuron.LayerNormalization(axis=node_axis, scale_and_bias=sb_enabled, eps=eps_val)
                
            param_shape = (1, *scale_val.shape)
            layer.init_parameters(param_shape)
            
            layer.gamma = scale_val.reshape(param_shape)
            layer.beta = bias_val.reshape(param_shape)
            
            pyaino_layers.append(layer)

        # --- InstanceNormalization の復元 ---
        elif op_type == 'InstanceNormalization':
            parts = node.name.split('__')
            sb_enabled = False
            for k in range(len(parts) - 1):
                if parts[k] == 'sb':
                    sb_enabled = (parts[k+1] == '1')
                    break
                    
            scale_name = node.input[1]
            bias_name = node.input[2]
            
            scale_val = initializers[scale_name]
            bias_val = initializers[bias_name]
            
            eps_val = 1e-12
            for attr in node.attribute:
                if attr.name == 'epsilon':
                    eps_val = attr.f
                    
            layer = Neuron.InstanceNorm2d(scale_and_bias=sb_enabled, eps=eps_val)
            
            C = len(scale_val)
            param_shape = (1, C, 1, 1)
            layer.init_parameters((1, C, 2, 2))
            
            layer.gamma = scale_val.reshape(param_shape)
            layer.beta = bias_val.reshape(param_shape)
            
            pyaino_layers.append(layer)

        # --- ScaleAndBias の復元 ---
        elif op_type == 'Mul' and "ScaleAndBias" in node.name:
            parts = node.name.split('__')
            axis_str = parts[3]
            if axis_str == 'None':
                axis_val = None
            else:
                axis_parts = axis_str.split('_')
                axis_val = tuple(int(x) for x in axis_parts)
                if len(axis_val) == 1:
                    axis_val = axis_val[0]
                    
            exclude_val = (parts[5] == '1')
            
            gamma_name = node.input[1]
            gamma_val = initializers[gamma_name]
            
            # 次の Add ノードから beta を取得
            next_node = graph.node[i+1]
            beta_name = next_node.input[1]
            beta_val = initializers[beta_name]
            
            layer = Neuron.ScaleAndBias(axis=axis_val, exclude=exclude_val)
            layer.init_parameters(gamma_val.shape)
            
            layer.gamma = gamma_val
            layer.beta = beta_val
            
            pyaino_layers.append(layer)
            skip_nodes.add(next_node.name)

        # --- Transpose の復元 ---
        elif op_type == 'Transpose':
            perm_val = (1, 0)
            for attr in node.attribute:
                if attr.name == 'perm':
                    perm_val = tuple(attr.ints)
            layer = Functions.Transpose(*perm_val)
            pyaino_layers.append(layer)

        # --- Reshape の復元 ---
        elif op_type == 'Reshape':
            shape_name = node.input[1]
            shape_val = initializers[shape_name]
            target_shape = tuple(int(x) for x in shape_val)
            pyaino_layers.append(Functions.Reshape(target_shape))

        # --- ReduceMean の復元 ---
        elif op_type == 'ReduceMean':
            keepdims_val = True
            for attr in node.attribute:
                if attr.name == 'keepdims':
                    keepdims_val = (attr.i == 1)
            
            axes_val = None
            if len(node.input) > 1:
                axes_name = node.input[1]
                if axes_name in initializers:
                    axes_arr = initializers[axes_name]
                    if axes_arr.ndim == 0:
                        axes_val = int(axes_arr)
                    elif len(axes_arr) == 1:
                        axes_val = int(axes_arr[0])
                    else:
                        axes_val = tuple(int(x) for x in axes_arr)
            
            pyaino_layers.append(Functions.Mean(axis=axes_val, keepdims=keepdims_val))

        # --- ConvTranspose (1D / 2D) の復元 ---
        elif op_type == 'ConvTranspose':
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            
            w_onnx = initializers[weight_name]
            b_onnx = initializers[bias_name] if bias_name else None
            
            strides = []
            pads = []
            kernel_shape = []
            for attr in node.attribute:
                if attr.name == 'strides':
                    strides = list(attr.ints)
                elif attr.name == 'pads':
                    pads = list(attr.ints)
                elif attr.name == 'kernel_shape':
                    kernel_shape = list(attr.ints)
                    
            bias_enabled = bias_name is not None
            
            if len(kernel_shape) == 1:
                C, M, Fw = w_onnx.shape
                stride = strides[0] if strides else 2
                pad = pads[0] if pads else 1
                layer = Neuron.Conv1dTransposeLayer(M, Fw, stride=stride, pad=pad, bias=bias_enabled)
                layer.config = (C, None, M, Fw, stride, pad, None)
                layer.parameters.init_parameter()
                w_pyaino = w_onnx.reshape(C, M * Fw)
                layer.parameters.w = w_pyaino
                if bias_enabled:
                    layer.parameters.b = b_onnx
                pyaino_layers.append(layer)
            else:
                C, M, Fh, Fw = w_onnx.shape
                Sh, Sw = strides if strides else (2, 2)
                pad = pads[0] if pads else 1
                layer = Neuron.Conv2dTransposeLayer(M, (Fh, Fw), stride=(Sh, Sw), pad=pad, bias=bias_enabled)
                layer.config = (C, None, None, M, Fh, Fw, Sh, Sw, pad, None, None)
                layer.parameters.init_parameter()
                w_pyaino = w_onnx.reshape(C, M * Fh * Fw)
                layer.parameters.w = w_pyaino
                if bias_enabled:
                    layer.parameters.b = b_onnx
                pyaino_layers.append(layer)

        # --- Resize (Interpolate2d) の復元 ---
        elif op_type == 'Resize':
            mode_val = 'nearest'
            for attr in node.attribute:
                if attr.name == 'mode':
                    mode_val = attr.s.decode('utf-8') if isinstance(attr.s, bytes) else attr.s
            mode = 'nearest' if mode_val == 'nearest' else 'bilinear'
            
            scale_factor = None
            size = None
            if len(node.input) > 2 and node.input[2] != "":
                scales_name = node.input[2]
                if scales_name in initializers:
                    scales_arr = initializers[scales_name]
                    if len(scales_arr) >= 4:
                        scale_factor = (float(scales_arr[2]), float(scales_arr[3]))
            if len(node.input) > 3 and node.input[3] != "":
                sizes_name = node.input[3]
                if sizes_name in initializers:
                    sizes_arr = initializers[sizes_name]
                    if len(sizes_arr) >= 4:
                        size = (int(sizes_arr[2]), int(sizes_arr[3]))
                        
            if scale_factor is not None:
                layer = Neuron.Interpolate2d(scale_factor=scale_factor, mode=mode)
            elif size is not None:
                layer = Neuron.Interpolate2d(size=size, mode=mode)
            else:
                layer = Neuron.Interpolate2d(scale_factor=2, mode=mode)
            pyaino_layers.append(layer)

        # --- Cast の復元 (スキップ) ---
        elif op_type == 'Cast':
            continue

        # --- Gather (Embedding) の復元 ---
        elif op_type == 'Gather':
            axis_val = 0
            for attr in node.attribute:
                if attr.name == 'axis':
                    axis_val = attr.i
            
            if axis_val == 0:
                weight_name = node.input[0]
                w_onnx = initializers[weight_name]
                vocab_size, wordvec_size = w_onnx.shape
                
                layer = Neuron.Embedding(vocab_size, wordvec_size)
                layer.parameters()
                layer.parameters.w = w_onnx.copy()
                pyaino_layers.append(layer)

        # --- Div + Softmax (Softmax2) の復元 ---
        elif op_type == 'Div':
            next_node = graph.node[i+1] if i+1 < len(graph.node) else None
            if next_node and next_node.op_type == 'Softmax':
                temp_name = node.input[1]
                temp_val = 1.0
                if temp_name in initializers:
                    temp_arr = initializers[temp_name]
                    temp_val = float(temp_arr.flatten()[0])
                
                layer = Activators.Softmax2(temperature=temp_val)
                pyaino_layers.append(layer)
                skip_nodes.add(next_node.name)
            else:
                pass

        # --- Greater + Cast (Step) の復元 ---
        elif op_type == 'Greater':
            next_node = graph.node[i+1] if i+1 < len(graph.node) else None
            if next_node and next_node.op_type == 'Cast':
                t_name = node.input[1]
                t_val = 0.0
                if t_name in initializers:
                    t_arr = initializers[t_name]
                    t_val = float(t_arr.flatten()[0])
                    
                layer = Activators.Step(t=t_val)
                pyaino_layers.append(layer)
                skip_nodes.add(next_node.name)

        else:
            print(f"サポートされていない演算ノード '{node.name}' (型: {op_type}) をスキップします。")
            
    # 再構築された全レイヤーをまとめて Sequential モデルにします
    sequential_model = Neuron.Sequential(*pyaino_layers)
    return sequential_model
