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
    
    # 複数レイヤーを内包する Sequential と、単一のレイヤーオブジェクトの両方に対応します
    if pyaino_model.__class__.__name__ == 'Sequential':
        layers = pyaino_model.layers
    else:
        layers = [pyaino_model]
        
    for i, layer in enumerate(layers):
        class_name = layer.__class__.__name__
        layer_output_name = f"layer_{i}_out"
        
        # --- 全結合層 (LinearLayer & NeuronLayer) の変換 ---
        if class_name in ('LinearLayer', 'NeuronLayer'):
            # layer.config から入力幅(m)とニューロン数(n)を取得します
            in_features, out_features = layer.config
            
            # まだパラメータが初期化されていない場合は、ダミー入力を通して初期化をトリガーします
            w_val = np.array(layer.parameters.w) if layer.parameters.w is not None else None
            b_val = np.array(layer.parameters.b) if layer.bias and layer.parameters.b is not None else None
            
            if w_val is None:
                # 順伝播を走らせて内部的に w と b を生成させます
                _ = layer.forward(dummy_input)
                w_val = np.array(layer.parameters.w)
                if layer.bias:
                    b_val = np.array(layer.parameters.b)
            
            w_name = f"layer_{i}_weight"
            # 【重要】ONNXのGemmオペレータは重み行列を [out_features, in_features] で要求するため、
            # pyainoの [in_features, out_features] の重みを転置（w_val.T）して登録します。
            w_init = helper.make_tensor(
                name=w_name,
                data_type=TensorProto.FLOAT,
                dims=w_val.T.shape,
                vals=w_val.T.flatten().tolist()
            )
            onnx_initializers.append(w_init)
            
            inputs = [current_input_name, w_name]
            
            # バイアスが存在する場合の処理
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
                
            # Gemm (General Matrix Multiplication) ノードを作成します (Y = alpha * A * B' + beta * C)
            # transB=1 を指定することで、上で転置した重み行列を再び転置して正規の計算を行います。
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
            # config から各設定値を取得します (C:チャネル数、M:フィルタ数、Fh/Fw:カーネルサイズ、Sh/Sw:ストライド、pad:パディング)
            C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = layer.config
            
            w_val = np.array(layer.parameters.w) if layer.parameters.w is not None else None
            if w_val is None:
                _ = layer.forward(dummy_input)
                w_val = np.array(layer.parameters.w)
            
            # 【重要】pyainoの畳み込み重みは2次元の(C * Fh * Fw, M)の形状ですが、
            # ONNX의 Convは4次元の (out_channels, in_channels, kernel_height, kernel_width) すなわち (M, C, Fh, Fw) を要求します。
            # そのため、転置した上でリシェイプします。
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
            
            # バイアスが存在する場合の処理
            if layer.bias:
                b_val = np.array(layer.parameters.b) if layer.parameters.b is not None else None
                if b_val is None:
                    b_val = np.zeros(M, dtype=np.float32)
                b_name = f"layer_{i}_bias"
                b_init = helper.make_tensor(
                    name=b_name,
                    data_type=TensorProto.FLOAT,
                    dims=b_val.shape,
                    vals=b_val.tolist()
                )
                onnx_initializers.append(b_init)
                inputs.append(b_name)
                
            # Convノードの作成 (パディング、ストライド、カーネルサイズをマッピングします)
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

        # --- 単体の活性化関数レイヤーの変換 ---
        elif class_name in ('ReLU', 'ReLU_bkup') or issubclass(layer.__class__, Activators.ReLU):
            relu_node = helper.make_node(
                op_type='Relu',
                inputs=[current_input_name],
                outputs=[layer_output_name],
                name=f"relu_{i}"
            )
            onnx_nodes.append(relu_node)
            current_input_name = layer_output_name
            
        elif class_name in ('LReLU', 'LReLU_bkup') or issubclass(layer.__class__, Activators.LReLU):
            # 【重要】pyainoではLeakyReLUの負の傾きは 'c'（デフォルト0.01）に格納されているため、これを読み取ります。
            slope = getattr(layer, 'c', 0.01)
            lrelu_node = helper.make_node(
                op_type='LeakyRelu',
                inputs=[current_input_name],
                outputs=[layer_output_name],
                name=f"lrelu_{i}",
                alpha=float(slope)  # ONNXの属性名は 'alpha' にマッピング
            )
            onnx_nodes.append(lrelu_node)
            current_input_name = layer_output_name
            
        elif class_name in ('Sigmoid',) or issubclass(layer.__class__, Activators.Sigmoid):
            sig_node = helper.make_node(
                op_type='Sigmoid',
                inputs=[current_input_name],
                outputs=[layer_output_name],
                name=f"sigmoid_{i}"
            )
            onnx_nodes.append(sig_node)
            current_input_name = layer_output_name
            
        elif class_name in ('Tanh',) or issubclass(layer.__class__, Activators.Tanh):
            tanh_node = helper.make_node(
                op_type='Tanh',
                inputs=[current_input_name],
                outputs=[layer_output_name],
                name=f"tanh_{i}"
            )
            onnx_nodes.append(tanh_node)
            current_input_name = layer_output_name
            
        elif class_name in ('Softmax',) or issubclass(layer.__class__, Activators.Softmax):
            softmax_node = helper.make_node(
                op_type='Softmax',
                inputs=[current_input_name],
                outputs=[layer_output_name],
                name=f"softmax_{i}",
                axis=-1
            )
            onnx_nodes.append(softmax_node)
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
                act_class = act.__class__.__name__
                act_output_name = f"layer_{i}_act_out"
                
                # 内包されている活性化関数の種類に応じて、ONNXのノードを追加作成します。
                # 逆変換（onnx_to_pyaino）時に元に戻せるよう、ノード名に "embedded" というキーワードを含めます。
                if act_class == 'ReLU':
                    act_node = helper.make_node(
                        op_type='Relu',
                        inputs=[current_input_name],
                        outputs=[act_output_name],
                        name=f"embedded_relu_{i}"
                    )
                elif act_class == 'LReLU':
                    slope = getattr(act, 'c', 0.01)
                    act_node = helper.make_node(
                        op_type='LeakyRelu',
                        inputs=[current_input_name],
                        outputs=[act_output_name],
                        name=f"embedded_lrelu_{i}",
                        alpha=float(slope)
                    )
                elif act_class == 'Sigmoid':
                    act_node = helper.make_node(
                        op_type='Sigmoid',
                        inputs=[current_input_name],
                        outputs=[act_output_name],
                        name=f"embedded_sigmoid_{i}"
                    )
                elif act_class == 'Tanh':
                    act_node = helper.make_node(
                        op_type='Tanh',
                        inputs=[current_input_name],
                        outputs=[act_output_name],
                        name=f"embedded_tanh_{i}"
                    )
                elif act_class == 'Softmax':
                    act_node = helper.make_node(
                        op_type='Softmax',
                        inputs=[current_input_name],
                        outputs=[act_output_name],
                        name=f"embedded_softmax_{i}",
                        axis=-1
                    )
                else:
                    raise NotImplementedError(f"内包活性化関数 '{act_class}' の変換はサポートされていません。")
                
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
    
    # ONNX Runtime等の幅広い推論エンジンで確実に動作するよう、安定版の Opset 15 を指定します。
    opset = helper.make_opsetid("", 15)
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
    
    # pyainoの出力とONNX Runtimeの出力の絶対誤差を計算
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
