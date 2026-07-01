import numpy as np
import onnx
from onnx import numpy_helper
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

def load_weights_to_pyaino_model(model, initializers):
    """
    ONNX の階層的初期化子名から属性パスを辿り、pyaino モデルのパラメータをロードします。
    """
    for init_name, val in initializers.items():
        parts = init_name.split('.')
        param_name = parts[-1]
        attr_path = parts[:-1]
        
        curr_obj = model
        failed = False
        for part in attr_path:
            if part.isdigit():
                idx = int(part)
                if isinstance(curr_obj, (list, tuple)):
                    curr_obj = curr_obj[idx]
                elif curr_obj.__class__.__name__ in ('Sequential', 'Sequential2', 'SequentialWithLoss'):
                    curr_obj = curr_obj.layers[idx]
                else:
                    failed = True
                    break
            else:
                if hasattr(curr_obj, part):
                    curr_obj = getattr(curr_obj, part)
                else:
                    failed = True
                    break
                    
        if failed:
            continue
            
        if param_name == 'w':
            layer = curr_obj.layer
            layer_class = layer.__class__.__name__
            if layer_class in ('Conv2dLayer', 'ConvLayer'):
                # ONNX: (M, C, Fh, Fw) -> pyaino: (C*Fh*Fw, M)
                M, C, Fh, Fw = val.shape
                val_pyaino = val.reshape(M, -1).T
                curr_obj.w = val_pyaino
            elif layer_class in ('Conv1dLayer',):
                # ONNX: (M, C, Fw) -> pyaino: (C*Fw, M)
                M, C, Fw = val.shape
                val_pyaino = val.reshape(M, -1).T
                curr_obj.w = val_pyaino
            elif layer_class in ('Conv2dTransposeLayer', 'DeConv2dLayer', 'DeConvLayer'):
                # ONNX: (C, M, Fh, Fw) -> pyaino: (C, M*Fh*Fw)
                C, M, Fh, Fw = val.shape
                val_pyaino = val.reshape(C, -1)
                curr_obj.w = val_pyaino
            elif layer_class in ('Conv1dTransposeLayer', 'DeConv1dLayer'):
                # ONNX: (C, M, Fw) -> pyaino: (C, M*Fw)
                C, M, Fw = val.shape
                val_pyaino = val.reshape(C, -1)
                curr_obj.w = val_pyaino
            elif layer_class in ('LinearLayer', 'NeuronLayer'):
                # ONNX: (out, in) -> pyaino: (in, out)
                curr_obj.w = val.T
            else:
                curr_obj.w = val.copy()
        elif param_name == 'b':
            curr_obj.b = val.copy()
        elif param_name in ('gamma', 'beta', 'mu_ppl', 'sigma_ppl'):
            attr_val = getattr(curr_obj, param_name, None)
            if attr_val is not None:
                target_shape = attr_val.shape
                setattr(curr_obj, param_name, val.reshape(target_shape))

def reconstruct_sequential(nodes, initializers, prefix):
    """
    指定されたプレフィックスを持つONNXノード群から pyaino レイヤーのリストを作成します。
    """
    pyaino_layers = []
    skip_nodes = set()
    
    for i, node in enumerate(nodes):
        #if not node.name.startswith(prefix): # 20260701AI
        if not (node.name.startswith(prefix) or node.name.startswith("embedded_" + prefix)):
            continue
        
        if node.name in skip_nodes:
            continue
            
        op_type = node.op_type
        
        if op_type == 'Gemm':
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            
            w_onnx = initializers[weight_name]
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
            
        elif op_type == 'Conv':
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            w_onnx = initializers[weight_name]
            b_onnx = initializers[bias_name] if bias_name else None
            
            strides, pads, kernel_shape = [], [], []
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
                layer.parameters.w = w_onnx.reshape(M, -1).T
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
                layer.parameters.w = w_onnx.reshape(M, -1).T
                if bias_enabled:
                    layer.parameters.b = b_onnx
                pyaino_layers.append(layer)
            
        elif op_type in ('Relu', 'LeakyRelu', 'Sigmoid', 'Tanh', 'Softmax', 'Identity', 'Elu', 'Softplus', 'Gelu', 'Swish', 'Mish'):
            if "embedded" in node.name:
                if len(pyaino_layers) > 0:
                    prev_layer = pyaino_layers[-1]
                    if hasattr(prev_layer, 'postphase'):
                        prev_layer.postphase.activator = restore_activation(op_type, node)
            else:
                pyaino_layers.append(restore_activation(op_type, node))
                
        elif op_type == 'Flatten':
            pyaino_layers.append(Neuron.Flatten())
            
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
                
        elif op_type == 'GlobalAveragePool':
            pyaino_layers.append(Neuron.GlobalAveragePooling())
            
        elif op_type == 'BatchNormalization':
            parts = node.name.split('__')
            orig_class_name = parts[0]
            sb_enabled = False
            for k in range(len(parts) - 1):
                if parts[k] == 'sb':
                    sb_enabled = (parts[k+1] == '1')
                    break
                    
            scale_name, bias_name, mean_name, var_name = node.input[1:5]
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
            
        elif op_type == 'LayerNormalization':
            parts = node.name.split('__')
            orig_class_name = parts[0]
            sb_enabled = False
            for k in range(len(parts) - 1):
                if parts[k] == 'sb':
                    sb_enabled = (parts[k+1] == '1')
                    break
                    
            scale_name, bias_name = node.input[1:3]
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
            
        elif op_type == 'InstanceNormalization':
            parts = node.name.split('__')
            sb_enabled = False
            for k in range(len(parts) - 1):
                if parts[k] == 'sb':
                    sb_enabled = (parts[k+1] == '1')
                    break
                    
            scale_name, bias_name = node.input[1:3]
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
            
            next_node = nodes[i+1] if i+1 < len(nodes) else None
            if next_node:
                beta_name = next_node.input[1]
                beta_val = initializers[beta_name]
                
                layer = Neuron.ScaleAndBias(axis=axis_val, exclude=exclude_val)
                layer.init_parameters(gamma_val.shape)
                layer.gamma = gamma_val
                layer.beta = beta_val
                pyaino_layers.append(layer)
                skip_nodes.add(next_node.name)
            
        elif op_type == 'Transpose':
            perm_val = (1, 0)
            for attr in node.attribute:
                if attr.name == 'perm':
                    perm_val = tuple(attr.ints)
            pyaino_layers.append(Functions.Transpose(*perm_val))
            
        elif op_type == 'Reshape':
            shape_name = node.input[1]
            shape_val = initializers[shape_name]
            target_shape = tuple(int(x) for x in shape_val)
            pyaino_layers.append(Functions.Reshape(target_shape))
            
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
            
        elif op_type == 'ConvTranspose':
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            w_onnx = initializers[weight_name]
            b_onnx = initializers[bias_name] if bias_name else None
            
            strides, pads, kernel_shape = [], [], []
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
                layer.parameters.w = w_onnx.reshape(C, M * Fw)
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
                layer.parameters.w = w_onnx.reshape(C, M * Fh * Fw)
                if bias_enabled:
                    layer.parameters.b = b_onnx
                pyaino_layers.append(layer)
                
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
                pyaino_layers.append(Neuron.Interpolate2d(scale_factor=scale_factor, mode=mode))
            elif size is not None:
                pyaino_layers.append(Neuron.Interpolate2d(size=size, mode=mode))
            else:
                pyaino_layers.append(Neuron.Interpolate2d(scale_factor=2, mode=mode))
                
        elif op_type == 'Cast':
            continue
            
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
                
        elif op_type == 'Div':
            next_node = nodes[i+1] if i+1 < len(nodes) else None
            if next_node and next_node.op_type == 'Softmax':
                temp_name = node.input[1]
                temp_val = 1.0
                if temp_name in initializers:
                    temp_val = float(initializers[temp_name].flatten()[0])
                pyaino_layers.append(Activators.Softmax2(temperature=temp_val))
                skip_nodes.add(next_node.name)
                
        elif op_type == 'Greater':
            next_node = nodes[i+1] if i+1 < len(nodes) else None
            if next_node and next_node.op_type == 'Cast':
                t_name = node.input[1]
                t_val = 0.0
                if t_name in initializers:
                    t_val = float(initializers[t_name].flatten()[0])
                pyaino_layers.append(Activators.Step(t=t_val))
                skip_nodes.add(next_node.name)
                
    return pyaino_layers

def import_onnx_to_pyaino(onnx_path):
    """
    ONNXモデルファイルを読み込み、pyainoのモデル（Sequential, UNet, CifarResNet, MyVAE）として再構築します。
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    
    # 構造の自動判定
    is_unet = False
    is_resnet = False
    is_vae = False
    
    unet_prefix = None
    for name in initializers.keys():
        if "model.core.down" in name or "model.core.stem" in name:
            is_unet = True
            unet_prefix = "model.core"
            break
        elif name.startswith("core.down.") or name.startswith("core.stem."):
            is_unet = True
            unet_prefix = "core"
            break
        elif "stages." in name or "stem." in name:
            is_resnet = True
            break
        elif "encoder." in name or "decoder." in name or "sampling." in name:
            is_vae = True
            break
            
    # === UNet の復元 ===
    if is_unet:
        depth = 0
        n_bottom = 0
        base_ch = 32
        in_ch = None
        bottleneck = False
        down_idx = 3 if unet_prefix == "model.core" else 2
        bot_idx = 3 if unet_prefix == "model.core" else 2
        
        for name in initializers.keys():
            if f"{unet_prefix}.down." in name:
                parts = name.split('.')
                idx = int(parts[down_idx])
                depth = max(depth, idx + 1)
            if f"{unet_prefix}.bot." in name:
                parts = name.split('.')
                idx = int(parts[bot_idx])
                n_bottom = max(n_bottom, idx + 1)
            if f"{unet_prefix}.down.0.convs.2" in name:
                bottleneck = True
                
        stem_w_name = f"{unet_prefix}.stem.parameters.w"
        if stem_w_name in initializers:
            base_ch = initializers[stem_w_name].shape[0]
            
        out_w_name = f"{unet_prefix}.out.parameters.w"
        if out_w_name in initializers:
            in_ch = initializers[out_w_name].shape[0]
            
        from pyaino.UNet import UNet
        unet = UNet(depth=depth, in_ch=in_ch, base_ch=base_ch, bottleneck=bottleneck, n_bottom=n_bottom)
        load_weights_to_pyaino_model(unet, initializers)
        return unet
        
    # === CifarResNet の復元 ===
    elif is_resnet:
        n_stage = 0
        base_ch = 16
        classes = 10
        
        for name in initializers.keys():
            if "stages." in name:
                parts = name.split('.')
                idx = int(parts[1])
                n_stage = max(n_stage, idx + 1)
                
        stem_w_name = "stem.parameters.w"
        if stem_w_name in initializers:
            base_ch = initializers[stem_w_name].shape[0]
            
        head_w_name = "head.net.1.parameters.w"
        if head_w_name in initializers:
            classes = initializers[head_w_name].shape[0]
            
        from pyaino.ResNet import CifarResNet
        resnet = CifarResNet(classes=classes, n_stage=n_stage, base_ch=base_ch)
        load_weights_to_pyaino_model(resnet, initializers)
        return resnet
        
    # === VAE (MyVAE) の復元 ===
    elif is_vae:
        encoder_layers = reconstruct_sequential(graph.node, initializers, "encoder.")
        decoder_layers = reconstruct_sequential(graph.node, initializers, "decoder.")
        
        encoder = Neuron.Sequential(*encoder_layers)
        decoder = Neuron.Sequential(*decoder_layers)
        
        # sampling.rate などの抽出
        rate_val = 1.0
        for name in initializers.keys():
            if "sampling.rate" in name:
                rate_val = float(initializers[name])
                break
                
        sampling = Neuron.LatentSampling(rate=rate_val)
        
        from pyaino import stems_blocks_heads as sbh
        from pyaino.LossFunctions import MeanSquaredError
        vae = sbh.MyVAE(encoder, sampling, decoder, MeanSquaredError())
        
        load_weights_to_pyaino_model(vae, initializers)
        return vae
        
    # === Sequential の復元 (従来のフォールバック) ===
    else:
        pyaino_layers = reconstruct_sequential(graph.node, initializers, "layers.")
        return Neuron.Sequential(*pyaino_layers)
