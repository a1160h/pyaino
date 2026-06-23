import numpy as np
import onnx
from onnx import helper, TensorProto
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

def make_activation_node(act_layer, input_name, output_name, name_prefix, onnx_nodes, onnx_initializers):
    act_class = act_layer.__class__.__name__
    if act_class in ('ReLU', 'ReLU_bkup') or issubclass(act_layer.__class__, Activators.ReLU):
        return helper.make_node('Relu', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__relu")
    elif act_class in ('LReLU', 'LReLU_bkup') or issubclass(act_layer.__class__, Activators.LReLU):
        slope = getattr(act_layer, 'c', 0.01)
        return helper.make_node('LeakyRelu', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__lrelu", alpha=float(slope))
    elif act_class in ('Sigmoid', 'SigmoidOut', 'SigmoidWithLoss') or issubclass(act_layer.__class__, (Activators.Sigmoid, Activators.SigmoidOut, Activators.SigmoidWithLoss)):
        return helper.make_node('Sigmoid', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__sigmoid")
    elif act_class in ('Tanh',) or issubclass(act_layer.__class__, Activators.Tanh):
        return helper.make_node('Tanh', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__tanh")
    elif act_class in ('Softmax', 'Softmax2', 'SoftmaxWithLoss', 'SoftmaxWithLoss2', 'SoftmaxCrossEntropy') or issubclass(act_layer.__class__, (Activators.Softmax, Activators.Softmax2, Activators.SoftmaxWithLoss, Activators.SoftmaxWithLoss2, Activators.SoftmaxCrossEntropy)):
        temp_val = getattr(act_layer, 'temperature', 1.0)
        if temp_val != 1.0:
            temp_name = f"{name_prefix}_softmax_temp"
            temp_init = helper.make_tensor(
                name=temp_name,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[float(temp_val)]
            )
            onnx_initializers.append(temp_init)
            
            div_out = f"{name_prefix}_softmax_div_out"
            div_node = helper.make_node(
                op_type='Div',
                inputs=[input_name, temp_name],
                outputs=[div_out],
                name=f"{name_prefix}__softmax_div"
            )
            onnx_nodes.append(div_node)
            softmax_in = div_out
        else:
            softmax_in = input_name
            
        return helper.make_node('Softmax', inputs=[softmax_in], outputs=[output_name], name=f"{name_prefix}__softmax", axis=-1)
    elif act_class in ('Identity',) or issubclass(act_layer.__class__, Activators.Identity):
        return helper.make_node('Identity', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__identity")
    elif act_class in ('ELU',) or issubclass(act_layer.__class__, Activators.ELU):
        c = getattr(act_layer, 'c', 1.0)
        return helper.make_node('Elu', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__elu", alpha=float(c))
    elif act_class in ('Softplus',) or issubclass(act_layer.__class__, Activators.Softplus):
        return helper.make_node('Softplus', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__softplus")
    elif act_class in ('GELU',) or issubclass(act_layer.__class__, Activators.GELU):
        return helper.make_node('Gelu', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__gelu", approximate='none')
    elif act_class in ('GELUap',) or issubclass(act_layer.__class__, Activators.GELUap):
        return helper.make_node('Gelu', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__geluap", approximate='tanh')
    elif act_class in ('Swish',) or issubclass(act_layer.__class__, Activators.Swish):
        beta = getattr(act_layer, 'beta', 1.0)
        return helper.make_node('Swish', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__swish", alpha=float(beta))
    elif act_class in ('Mish',) or issubclass(act_layer.__class__, Activators.Mish):
        return helper.make_node('Mish', inputs=[input_name], outputs=[output_name], name=f"{name_prefix}__mish")
    elif act_class in ('Step',) or issubclass(act_layer.__class__, Activators.Step):
        t_val = getattr(act_layer, 't', 0.0)
        t_name = f"{name_prefix}_step_t"
        t_init = helper.make_tensor(
            name=t_name,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[float(t_val)]
        )
        onnx_initializers.append(t_init)
        
        greater_out = f"{name_prefix}_step_greater_out"
        greater_node = helper.make_node(
            op_type='Greater',
            inputs=[input_name, t_name],
            outputs=[greater_out],
            name=f"{name_prefix}__step_greater"
        )
        onnx_nodes.append(greater_node)
        
        return helper.make_node('Cast', inputs=[greater_out], outputs=[output_name], name=f"{name_prefix}__step_cast", to=TensorProto.FLOAT)
    else:
        raise NotImplementedError(f"活性化関数 '{act_class}' の変換はサポートされていません。")


def flatten_layers(pyaino_model_or_layer):
    """
    Sequential モデルの中にさらに Sequential がネストされている場合、
    それを再帰的に平坦化して、フラットなレイヤーのリストを返します。
    """
    flat_layers = []
    class_name = pyaino_model_or_layer.__class__.__name__
    
    if class_name in ('Sequential', 'Sequential2', 'SequentialWithLoss'):
        for layer in pyaino_model_or_layer.layers:
            flat_layers.extend(flatten_layers(layer))
    else:
        flat_layers.append(pyaino_model_or_layer)
        
    return flat_layers

def export_pyaino_to_onnx(pyaino_model, dummy_input, output_path="model.onnx"):
    """
    pyainoのモデル（Sequentialまたは単一レイヤー）を直接ONNX形式へエクスポートします。
    PyTorchには依存せず、公式のonnx Pythonライブラリのみを使用します。
    
    引数:
        pyaino_model: 変換対象 of pyainoモデルオブジェクト
        dummy_input: 入力の形状（Shape）を確定させるためのダミー入力データ（NumPy配列）
        output_path: 保存するONNXファイルのパス
    """
    onnx_nodes = []          # ONNXノード（計算処理）のリスト
    onnx_initializers = []     # ONNX初期化子（重みやバイアスの定数）のリスト
    
    # 1. グラフ全体の入力情報を定義
    input_shape = list(dummy_input.shape)
    # バッチサイズ次元（1番目の次元）を動的なバッチ処理に対応するため 'batch_size' 文字列に設定します
    input_shape[0] = 'batch_size'
    
    current_input_name = "input_tensor"
    input_value_info = helper.make_tensor_value_info(
        name=current_input_name,
        elem_type=TensorProto.FLOAT,
        shape=input_shape
    )
    
    # レイヤーリストを取得（ネストされた Sequential も再帰的に平坦化）
    layers = flatten_layers(pyaino_model)
        
    current_dummy = dummy_input.copy()
    for i, layer in enumerate(layers):
        class_name = layer.__class__.__name__
        layer_output_name = f"layer_{i}_out"
        
        in_shape = current_dummy.shape
        # 順伝播を走らせて出力を得るとともに、パラメータ初期化を確実に行います
        current_dummy = layer.forward(current_dummy)
        
        # --- 全結合層 (LinearLayer & NeuronLayer) の変換 ---
        if class_name in ('LinearLayer', 'NeuronLayer'):
            in_features, out_features = layer.config
            w_val = np.array(layer.parameters.w)
            b_val = np.array(layer.parameters.b) if layer.bias else None
            
            # 入力テンソルが3次元以上（例: 畳み込みやプーリング層の出力）の場合、ONNXのGemm前にFlattenノードを挿入します
            if len(in_shape) > 2:
                flatten_output_name = f"layer_{i}_flatten"
                flatten_node = helper.make_node(
                    op_type='Flatten',
                    inputs=[current_input_name],
                    outputs=[flatten_output_name],
                    name=f"flatten_{i}",
                    axis=1
                )
                onnx_nodes.append(flatten_node)
                current_input_name = flatten_output_name
            
            w_name = f"layer_{i}_weight"
            w_init = helper.make_tensor(
                name=w_name,
                data_type=TensorProto.FLOAT,
                dims=w_val.T.shape,
                vals=w_val.T.flatten().tolist()
            )
            onnx_initializers.append(w_init)
            
            inputs = [current_input_name, w_name]
            
            if layer.bias:
                b_name = f"layer_{i}_bias"
                b_init = helper.make_tensor(
                    name=b_name,
                    data_type=TensorProto.FLOAT,
                    dims=b_val.shape,
                    vals=b_val.tolist()
                )
                onnx_initializers.append(b_init)
                inputs.append(b_name)
                
            gemm_node = helper.make_node(
                op_type='Gemm',
                inputs=inputs,
                outputs=[layer_output_name],
                name=f"gemm_{i}",
                alpha=1.0,
                beta=1.0 if layer.bias else 0.0,
                transB=1
            )
            onnx_nodes.append(gemm_node)
            current_input_name = layer_output_name
            
        # --- 2次元畳み込み層 (Conv2dLayer) の変換 ---
        elif class_name in ('Conv2dLayer', 'ConvLayer'):
            C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = layer.config
            w_val = np.array(layer.parameters.w)
            
            w_onnx = w_val.T.reshape(M, C, Fh, Fw)
            w_name = f"layer_{i}_weight"
            w_init = helper.make_tensor(
                name=w_name,
                data_type=TensorProto.FLOAT,
                dims=w_onnx.shape,
                vals=w_onnx.flatten().tolist()
            )
            onnx_initializers.append(w_init)
            
            inputs = [current_input_name, w_name]
            
            if layer.bias:
                b_val = np.array(layer.parameters.b)
                b_name = f"layer_{i}_bias"
                b_init = helper.make_tensor(
                    name=b_name,
                    data_type=TensorProto.FLOAT,
                    dims=b_val.shape,
                    vals=b_val.tolist()
                )
                onnx_initializers.append(b_init)
                inputs.append(b_name)
                
            conv_node = helper.make_node(
                op_type='Conv',
                inputs=inputs,
                outputs=[layer_output_name],
                name=f"conv_{i}",
                pads=[pad, pad, pad, pad],
                strides=[Sh, Sw],
                kernel_shape=[Fh, Fw]
            )
            onnx_nodes.append(conv_node)
            current_input_name = layer_output_name

        # --- 2次元プーリング層 (Pooling2dLayer) の変換 ---
        elif class_name in ('Pooling2dLayer', 'PoolingLayer'):
            C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = layer.config
            op_type = 'MaxPool' if getattr(layer, 'method', 'max') == 'max' else 'AveragePool'
            pool_node = helper.make_node(
                op_type=op_type,
                inputs=[current_input_name],
                outputs=[layer_output_name],
                name=f"pooling_{i}",
                kernel_shape=[pool_h, pool_w],
                strides=[pool_h, pool_w],
                pads=[pad, pad, pad, pad]
            )
            onnx_nodes.append(pool_node)
            current_input_name = layer_output_name

        # --- Dropout / Dropout2 の変換 ---
        elif class_name in ('Dropout', 'Dropout2'):
            preset_val = getattr(layer, 'preset', None)
            ratio_name = f"layer_{i}_dropout_ratio"
            ratio_val = float(preset_val) if preset_val is not None else 0.5
            ratio_init = helper.make_tensor(
                name=ratio_name,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[ratio_val]
            )
            onnx_initializers.append(ratio_init)
            
            train_mode_name = f"layer_{i}_dropout_train_mode"
            train_mode_init = helper.make_tensor(
                name=train_mode_name,
                data_type=TensorProto.BOOL,
                dims=[],
                vals=[False]
            )
            onnx_initializers.append(train_mode_init)
            
            dropout_node = helper.make_node(
                op_type='Dropout',
                inputs=[current_input_name, ratio_name, train_mode_name],
                outputs=[layer_output_name],
                name=f"{class_name}__{i}__preset__{ratio_val}"
            )
            onnx_nodes.append(dropout_node)
            current_input_name = layer_output_name

        # --- GlobalAveragePooling の変換 ---
        elif class_name == 'GlobalAveragePooling':
            gap_out = f"layer_{i}_gap_out"
            gap_node = helper.make_node(
                op_type='GlobalAveragePool',
                inputs=[current_input_name],
                outputs=[gap_out],
                name=f"GlobalAveragePool__{i}"
            )
            onnx_nodes.append(gap_node)
            
            flatten_node = helper.make_node(
                op_type='Flatten',
                inputs=[gap_out],
                outputs=[layer_output_name],
                name=f"GlobalAveragePool_flatten__{i}",
                axis=1
            )
            onnx_nodes.append(flatten_node)
            current_input_name = layer_output_name

        # --- BatchNormalization / BatchNorm1d / BatchNorm2d の変換 ---
        elif class_name in ('BatchNormalization', 'batch_normalization', 'BatchNorm1d', 'batch_norm_1d', 'BatchNorm2d', 'batch_norm_2d'):
            C = layer.gamma.shape[1] if layer.gamma is not None else (layer.mu_ppl.shape[1] if layer.mu_ppl is not None else 1)
            
            if layer.sb and layer.gamma is not None:
                gamma_val = layer.gamma[0].flatten().tolist()
                beta_val = layer.beta[0].flatten().tolist()
            else:
                gamma_val = [1.0] * C
                beta_val = [0.0] * C
                
            if layer.ppl and layer.mu_ppl is not None:
                mean_val = layer.mu_ppl[0].flatten().tolist()
                var_val = (layer.sigma_ppl[0].flatten() ** 2).tolist()
            else:
                mean_val = [0.0] * C
                var_val = [1.0] * C
                
            scale_name = f"layer_{i}_bn_scale"
            scale_init = helper.make_tensor(scale_name, TensorProto.FLOAT, [C], gamma_val)
            onnx_initializers.append(scale_init)
            
            bias_name = f"layer_{i}_bn_bias"
            bias_init = helper.make_tensor(bias_name, TensorProto.FLOAT, [C], beta_val)
            onnx_initializers.append(bias_init)
            
            mean_name = f"layer_{i}_bn_mean"
            mean_init = helper.make_tensor(mean_name, TensorProto.FLOAT, [C], mean_val)
            onnx_initializers.append(mean_init)
            
            var_name = f"layer_{i}_bn_var"
            var_init = helper.make_tensor(var_name, TensorProto.FLOAT, [C], var_val)
            onnx_initializers.append(var_init)
            
            bn_node = helper.make_node(
                op_type='BatchNormalization',
                inputs=[current_input_name, scale_name, bias_name, mean_name, var_name],
                outputs=[layer_output_name],
                name=f"{class_name}__{i}__sb__{1 if layer.sb else 0}",
                epsilon=float(layer.eps)
            )
            onnx_nodes.append(bn_node)
            current_input_name = layer_output_name

        # --- LayerNormalization / LayerNorm1d / LayerNorm2d の変換 ---
        elif class_name in ('LayerNormalization', 'LayerNorm1d', 'LayerNorm2d'):
            if layer.gamma is not None:
                param_shape = list(layer.gamma.shape[1:])
            else:
                param_shape = [current_dummy.shape[-1]]
                
            param_size = int(np.prod(param_shape))
            
            if layer.sb and layer.gamma is not None:
                gamma_val = layer.gamma[0].flatten().tolist()
                beta_val = layer.beta[0].flatten().tolist()
            else:
                gamma_val = [1.0] * param_size
                beta_val = [0.0] * param_size
                
            scale_name = f"layer_{i}_ln_scale"
            scale_init = helper.make_tensor(scale_name, TensorProto.FLOAT, param_shape, gamma_val)
            onnx_initializers.append(scale_init)
            
            bias_name = f"layer_{i}_ln_bias"
            bias_init = helper.make_tensor(bias_name, TensorProto.FLOAT, param_shape, beta_val)
            onnx_initializers.append(bias_init)
            
            if class_name == 'LayerNorm2d':
                ln_axis = -3
            elif class_name == 'LayerNorm1d':
                ln_axis = -2
            else:
                ln_axis = -1
                
            ln_node = helper.make_node(
                op_type='LayerNormalization',
                inputs=[current_input_name, scale_name, bias_name],
                outputs=[layer_output_name],
                name=f"{class_name}__{i}__sb__{1 if layer.sb else 0}",
                axis=ln_axis,
                epsilon=float(layer.eps)
            )
            onnx_nodes.append(ln_node)
            current_input_name = layer_output_name

        # --- InstanceNorm2d の変換 ---
        elif class_name == 'InstanceNorm2d':
            C = layer.gamma.shape[1] if layer.gamma is not None else 1
            if layer.sb and layer.gamma is not None:
                gamma_val = layer.gamma[0].flatten().tolist()
                beta_val = layer.beta[0].flatten().tolist()
            else:
                gamma_val = [1.0] * C
                beta_val = [0.0] * C
                
            scale_name = f"layer_{i}_in_scale"
            scale_init = helper.make_tensor(scale_name, TensorProto.FLOAT, [C], gamma_val)
            onnx_initializers.append(scale_init)
            
            bias_name = f"layer_{i}_in_bias"
            bias_init = helper.make_tensor(bias_name, TensorProto.FLOAT, [C], beta_val)
            onnx_initializers.append(bias_init)
            
            in_node = helper.make_node(
                op_type='InstanceNormalization',
                inputs=[current_input_name, scale_name, bias_name],
                outputs=[layer_output_name],
                name=f"InstanceNorm2d__{i}__sb__{1 if layer.sb else 0}",
                epsilon=float(layer.eps)
            )
            onnx_nodes.append(in_node)
            current_input_name = layer_output_name

        # --- ScaleAndBias の変換 ---
        elif class_name == 'ScaleAndBias':
            gamma_shape = list(layer.gamma.shape)
            beta_shape = list(layer.beta.shape)
            gamma_val = layer.gamma.flatten().tolist()
            beta_val = layer.beta.flatten().tolist()
            
            gamma_name = f"layer_{i}_sb_gamma"
            gamma_init = helper.make_tensor(gamma_name, TensorProto.FLOAT, gamma_shape, gamma_val)
            onnx_initializers.append(gamma_init)
            
            beta_name = f"layer_{i}_sb_beta"
            beta_init = helper.make_tensor(beta_name, TensorProto.FLOAT, beta_shape, beta_val)
            onnx_initializers.append(beta_init)
            
            if layer.axis is None:
                axis_str = "None"
            elif isinstance(layer.axis, (list, tuple)):
                axis_str = "_".join(str(x) for x in layer.axis)
            else:
                axis_str = str(layer.axis)
                
            mul_out = f"layer_{i}_sb_mul_out"
            mul_node = helper.make_node(
                op_type='Mul',
                inputs=[current_input_name, gamma_name],
                outputs=[mul_out],
                name=f"ScaleAndBias__{i}__axis__{axis_str}__ex__{1 if layer.exclude else 0}__mul"
            )
            onnx_nodes.append(mul_node)
            
            add_node = helper.make_node(
                op_type='Add',
                inputs=[mul_out, beta_name],
                outputs=[layer_output_name],
                name=f"ScaleAndBias__{i}__axis__{axis_str}__ex__{1 if layer.exclude else 0}__add"
            )
            onnx_nodes.append(add_node)
            current_input_name = layer_output_name

        # --- Reshape の変換 ---
        elif class_name == 'Reshape':
            shape_name = f"layer_{i}_reshape_shape"
            shape_val = list(layer.shape)
            shape_init = helper.make_tensor(
                name=shape_name,
                data_type=TensorProto.INT64,
                dims=[len(shape_val)],
                vals=shape_val
            )
            onnx_initializers.append(shape_init)
            
            reshape_node = helper.make_node(
                op_type='Reshape',
                inputs=[current_input_name, shape_name],
                outputs=[layer_output_name],
                name=f"Reshape__{i}"
            )
            onnx_nodes.append(reshape_node)
            current_input_name = layer_output_name

        # --- Mean の変換 (ONNX ReduceMean にマップ) ---
        elif class_name == 'Mean':
            axes_name = f"layer_{i}_mean_axes"
            axes_val = list(layer.axis) if isinstance(layer.axis, (list, tuple)) else ([layer.axis] if layer.axis is not None else [])
            inputs = [current_input_name]
            if layer.axis is not None:
                axes_init = helper.make_tensor(
                    name=axes_name,
                    data_type=TensorProto.INT64,
                    dims=[len(axes_val)],
                    vals=axes_val
                )
                onnx_initializers.append(axes_init)
                inputs.append(axes_name)
                
            mean_node = helper.make_node(
                op_type='ReduceMean',
                inputs=inputs,
                outputs=[layer_output_name],
                name=f"Mean__{i}",
                keepdims=1 if layer.keepdims else 0
            )
            onnx_nodes.append(mean_node)
            current_input_name = layer_output_name
            
        # --- Conv1dLayer の変換 ---
        elif class_name == 'Conv1dLayer':
            C, Iw, M, Fw, stride, pad, Ow = layer.config
            w_val = np.array(layer.parameters.w)
            w_onnx = w_val.T.reshape(M, C, Fw)
            
            w_name = f"layer_{i}_weight"
            w_init = helper.make_tensor(w_name, TensorProto.FLOAT, w_onnx.shape, w_onnx.flatten().tolist())
            onnx_initializers.append(w_init)
            
            inputs = [current_input_name, w_name]
            if layer.bias:
                b_val = np.array(layer.parameters.b)
                b_name = f"layer_{i}_bias"
                b_init = helper.make_tensor(b_name, TensorProto.FLOAT, b_val.shape, b_val.tolist())
                onnx_initializers.append(b_init)
                inputs.append(b_name)
                
            conv_node = helper.make_node(
                op_type='Conv',
                inputs=inputs,
                outputs=[layer_output_name],
                name=f"Conv1dLayer__{i}",
                pads=[pad, pad],
                strides=[stride],
                kernel_shape=[Fw]
            )
            onnx_nodes.append(conv_node)
            current_input_name = layer_output_name

        # --- Conv1dTransposeLayer / DeConv1dLayer の変換 ---
        elif class_name in ('Conv1dTransposeLayer', 'DeConv1dLayer'):
            C, Iw, M, Fw, stride, pad, Ow = layer.config
            w_val = np.array(layer.parameters.w)
            w_onnx = w_val.reshape(C, M, Fw)
            
            w_name = f"layer_{i}_weight"
            w_init = helper.make_tensor(w_name, TensorProto.FLOAT, w_onnx.shape, w_onnx.flatten().tolist())
            onnx_initializers.append(w_init)
            
            inputs = [current_input_name, w_name]
            if layer.bias:
                b_val = np.array(layer.parameters.b)
                if len(b_val) == M * Fw:
                    b_val = b_val.reshape(M, Fw).mean(axis=1)
                b_name = f"layer_{i}_bias"
                b_init = helper.make_tensor(b_name, TensorProto.FLOAT, [M], b_val.tolist())
                onnx_initializers.append(b_init)
                inputs.append(b_name)
                
            deconv_node = helper.make_node(
                op_type='ConvTranspose',
                inputs=inputs,
                outputs=[layer_output_name],
                name=f"{class_name}__{i}",
                pads=[pad, pad],
                strides=[stride],
                kernel_shape=[Fw]
            )
            onnx_nodes.append(deconv_node)
            current_input_name = layer_output_name

        # --- Conv2dTransposeLayer / DeConv2dLayer / DeConvLayer の変換 ---
        elif class_name in ('Conv2dTransposeLayer', 'DeConv2dLayer', 'DeConvLayer'):
            C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = layer.config
            w_val = np.array(layer.parameters.w)
            w_onnx = w_val.reshape(C, M, Fh, Fw)
            
            w_name = f"layer_{i}_weight"
            w_init = helper.make_tensor(w_name, TensorProto.FLOAT, w_onnx.shape, w_onnx.flatten().tolist())
            onnx_initializers.append(w_init)
            
            inputs = [current_input_name, w_name]
            if layer.bias:
                b_val = np.array(layer.parameters.b)
                if len(b_val) == M * Fh * Fw:
                    b_val = b_val.reshape(M, Fh, Fw).mean(axis=(1, 2))
                b_name = f"layer_{i}_bias"
                b_init = helper.make_tensor(b_name, TensorProto.FLOAT, [M], b_val.tolist())
                onnx_initializers.append(b_init)
                inputs.append(b_name)
                
            deconv_node = helper.make_node(
                op_type='ConvTranspose',
                inputs=inputs,
                outputs=[layer_output_name],
                name=f"{class_name}__{i}",
                pads=[pad, pad, pad, pad],
                strides=[Sh, Sw],
                kernel_shape=[Fh, Fw]
            )
            onnx_nodes.append(deconv_node)
            current_input_name = layer_output_name

        # --- 1次元プーリング層 (Pooling1dLayer) の変換 ---
        elif class_name in ('Pooling1dLayer', 'Pooling1d'):
            C, Iw, pool, pad, Ow = layer.config
            op_type = 'MaxPool' if getattr(layer, 'method', 'max') == 'max' else 'AveragePool'
            pool_node = helper.make_node(
                op_type=op_type,
                inputs=[current_input_name],
                outputs=[layer_output_name],
                name=f"{class_name}__{i}",
                kernel_shape=[pool],
                strides=[pool],
                pads=[pad, pad]
            )
            onnx_nodes.append(pool_node)
            current_input_name = layer_output_name

        # --- Interpolate2d / Interpolate2dLayer / Interpolate2dNearest の変換 ---
        elif class_name in ('Interpolate2d', 'Interpolate2dLayer', 'Interpolate2dNearest'):
            Ih, Iw, Oh, Ow, mode, align = layer.config
            
            scales_name = f"layer_{i}_resize_scales"
            scales_val = [1.0, 1.0, float(Oh)/float(Ih), float(Ow)/float(Iw)]
            scales_init = helper.make_tensor(
                name=scales_name,
                data_type=TensorProto.FLOAT,
                dims=[4],
                vals=scales_val
            )
            onnx_initializers.append(scales_init)
            
            resize_node = helper.make_node(
                op_type='Resize',
                inputs=[current_input_name, "", scales_name],
                outputs=[layer_output_name],
                name=f"resize_{i}",
                mode='nearest' if mode == 'nearest' else 'linear',
                coordinate_transformation_mode='asymmetric'
            )
            onnx_nodes.append(resize_node)
            current_input_name = layer_output_name

        # --- Embedding の変換 ---
        elif class_name == 'Embedding':
            vocab_size, wordvec_size = layer.config
            w_val = np.array(layer.parameters.w)
            
            w_name = f"layer_{i}_weight"
            w_init = helper.make_tensor(
                name=w_name,
                data_type=TensorProto.FLOAT,
                dims=w_val.shape,
                vals=w_val.flatten().tolist()
            )
            onnx_initializers.append(w_init)
            
            cast_out = f"layer_{i}_embed_cast_out"
            cast_node = helper.make_node(
                op_type='Cast',
                inputs=[current_input_name],
                outputs=[cast_out],
                name=f"Cast_for_Embedding__{i}",
                to=TensorProto.INT64
            )
            onnx_nodes.append(cast_node)
            
            gather_node = helper.make_node(
                op_type='Gather',
                inputs=[w_name, cast_out],
                outputs=[layer_output_name],
                name=f"Embedding__{i}",
                axis=0
            )
            onnx_nodes.append(gather_node)
            current_input_name = layer_output_name

        # --- Flatten の変換 ---
        elif class_name == 'Flatten':
            flatten_node = helper.make_node(
                op_type='Flatten',
                inputs=[current_input_name],
                outputs=[layer_output_name],
                name=f"Flatten__{i}",
                axis=1
            )
            onnx_nodes.append(flatten_node)
            current_input_name = layer_output_name

        # --- Transpose の変換 ---
        elif class_name == 'Transpose':
            axes_val = list(layer.axes)
            transpose_node = helper.make_node(
                op_type='Transpose',
                inputs=[current_input_name],
                outputs=[layer_output_name],
                name=f"Transpose__{i}",
                perm=axes_val
            )
            onnx_nodes.append(transpose_node)
            current_input_name = layer_output_name

        # --- 単体の活性化関数レイヤーの変換 ---
        elif class_name in ('ReLU', 'ReLU_bkup', 'LReLU', 'LReLU_bkup', 'Sigmoid', 'SigmoidOut', 'SigmoidWithLoss', 'Tanh', 'Softmax', 'Softmax2', 'SoftmaxWithLoss', 'SoftmaxWithLoss2', 'SoftmaxCrossEntropy', 'Identity', 'ELU', 'Softplus', 'GELU', 'GELUap', 'Swish', 'Mish', 'Step') or issubclass(layer.__class__, (Activators.ReLU, Activators.LReLU, Activators.Sigmoid, Activators.SigmoidOut, Activators.SigmoidWithLoss, Activators.Tanh, Activators.Softmax, Activators.Softmax2, Activators.SoftmaxWithLoss, Activators.SoftmaxWithLoss2, Activators.SoftmaxCrossEntropy, Activators.Identity, Activators.ELU, Activators.Softplus, Activators.GELU, Activators.GELUap, Activators.Swish, Activators.Mish, Activators.Step)):
            act_node = make_activation_node(layer, current_input_name, layer_output_name, f"layer_{i}", onnx_nodes, onnx_initializers)
            onnx_nodes.append(act_node)
            current_input_name = layer_output_name
            
        else:
            raise NotImplementedError(f"レイヤークラス '{class_name}' の直接ONNX変換は未実装です。")
            
        # --- 【超重要】レイヤーに内包されている活性化関数（postphase）の抽出処理 ---
        # pyainoのBaseLayer系列のクラスは、レイヤー内部のpostphaseに活性化関数を埋め込む設計になっています。
        # ONNXはこれを単一の計算ノードに分離しなければならないため、ここで別個のノードとして追加します。
        if hasattr(layer, 'postphase'):
            post = layer.postphase
            if hasattr(post, 'activator') and post.activator is not None:
                act = post.activator
                act_output_name = f"layer_{i}_act_out"
                
                # 内包されている活性化関数の種類に応じて、ONNXのノードを追加作成します。
                # 逆変換（onnx_to_pyaino）時に元に戻せるよう、ノード名に "embedded" というキーワードを含めます。
                act_node = make_activation_node(act, current_input_name, act_output_name, f"embedded_layer_{i}", onnx_nodes, onnx_initializers)
                
                onnx_nodes.append(act_node)
                current_input_name = act_output_name

    # 4. グラフ全体の出力情報を定義
    # ダミーデータをモデルに通し、最終出力の形状を自動取得します。
    y_dummy = pyaino_model.forward(dummy_input)
    output_shape = list(y_dummy.shape)
    output_shape[0] = 'batch_size'
    
    output_value_info = helper.make_tensor_value_info(
        name=current_input_name,
        elem_type=TensorProto.FLOAT,
        shape=output_shape
    )
    
    # 5. ONNXグラフとモデルの作成
    graph = helper.make_graph(
        nodes=onnx_nodes,
        name="pyaino_direct_onnx",
        inputs=[input_value_info],
        outputs=[output_value_info],
        initializer=onnx_initializers
    )
    
    # ONNX Runtime等の幅広い推論エンジンで確実に動作するよう、安定版の Opset 24 を指定します。
    opset = helper.make_opsetid("", 24)
    onnx_model = helper.make_model(graph, producer_name="pyaino_direct_converter", opset_imports=[opset])
    
    # モデルの整合性を自動検証し、問題なければディスクに保存します。
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path)
    
    print(f"[SUCCESS] ONNXモデルファイルが直接出力されました: {output_path}")
    return onnx_model

# --- 単体実行時の検証テストセクション ---
if __name__ == '__main__':
    print("--------------------------------------------------")
    print("pyaino -> ONNX 直接変換検証テストを実行中...")
    print("--------------------------------------------------")
    
    # テスト用の3層構成モデルを構築
    test_model = Neuron.Sequential(
        Neuron.LinearLayer(10, 32),               # 10次元から32次元の全結合層
        Activators.ReLU(),                         # 単体のReLU活性化レイヤー
        Neuron.NeuronLayer(32, 5, activate='Softmax') # 32次元から5次元の、Softmaxを内包したニューロン層
    )
    
    # テスト用のダミー入力データ (サイズ: 4x10) を用意
    dummy_x = np.random.randn(4, 10).astype(np.float32)
    
    # pyaino上での「正解」出力を計算
    pyaino_output = test_model.forward(dummy_x)
    print(f"pyaino 出力形状: {pyaino_output.shape}")
    
    # 直接ONNXへの変換を実行
    onnx_file_path = "verification_model.onnx"
    export_pyaino_to_onnx(test_model, dummy_x, onnx_file_path)
    
    # エクスポートしたONNXファイルをONNX Runtimeで読み込み検証
    print("ONNX Runtimeの推論セッションを初期化中...")
    ort_session = ort.InferenceSession(onnx_file_path)
    
    # ONNXの計算を実行
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_x}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_output = ort_outputs[0]
    
    print(f"ONNX Runtime 出力形状: {ort_output.shape}")
    
    # pyaino의 출력とONNX Runtimeの出力の絶対誤差を計算
    diff = np.abs(pyaino_output - ort_output)
    max_difference = np.max(diff)
    print(f"pyaino と ONNX Runtime の最大絶対誤差: {max_difference:.2e}")
    print("pyaino 出力結果:\n", pyaino_output)
    print("ONNX Runtime 出力結果:\n", ort_output)
    
    # 誤差が float32 の限界値（1e-6）以下ならテスト合格とします
    if max_difference < 1e-6:
        print("[PASS] 数値検証成功！ ONNXモデルの計算結果は pyaino と完全に一致しています。")
    else:
        print("[FAIL] 計算結果に許容値を超える不一致が検出されました。")
        sys.exit(1)
