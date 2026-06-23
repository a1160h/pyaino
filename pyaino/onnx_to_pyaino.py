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
    
    for i, node in enumerate(graph.node):
        op_type = node.op_type
        
        # --- 全結合層 (Gemm) ノードの復元 ---
        # ※ ONNXにおける Gemm ノードは、pyainoの NeuronLayer にマッピングします。
        # NeuronLayerはBaseLayerを継承しているため、後段の活性化関数埋め込み(postphase)が扱えます。
        if op_type == 'Gemm':
            # Gemmの入力順は [入力テンソル名, 重みテンソル名, バイアステンソル名(オプション)] です
            input_name = node.input[0]
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            
            w_onnx = initializers[weight_name]  # ONNXでの形状: (out_features, in_features)
            b_onnx = initializers[bias_name] if bias_name else None
            
            out_features, in_features = w_onnx.shape
            bias_enabled = bias_name is not None
            
            # NeuronLayerをインスタンス化します
            layer = Neuron.NeuronLayer(in_features, out_features, bias=bias_enabled)
            
            # pyainoの初期パラメータ設定を確定させます
            layer.config = (in_features, out_features)
            layer.parameters.init_parameter()
            
            # 【重要】ONNXの重み [out, in] を pyainoの [in, out] レイアウトに戻すため転置（.T）して流し込みます。
            layer.parameters.w = w_onnx.T
            if bias_enabled:
                layer.parameters.b = b_onnx
                
            pyaino_layers.append(layer)
            
        # --- 2次元畳み込み層 (Conv) ノードの復元 ---
        elif op_type == 'Conv':
            # Convの入力は [入力テンソル名, 重みテンソル名, バイアステンソル名(オプション)] です
            input_name = node.input[0]
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 else None
            
            w_onnx = initializers[weight_name]  # ONNXでの形状: (M, C, Fh, Fw)
            b_onnx = initializers[bias_name] if bias_name else None
            
            M, C, Fh, Fw = w_onnx.shape
            
            # ストライドとパディングの属性をスキャンして取得します
            strides = [1, 1]
            pads = [0, 0, 0, 0]
            for attr in node.attribute:
                if attr.name == 'strides':
                    strides = list(attr.ints)
                elif attr.name == 'pads':
                    pads = list(attr.ints)
                    
            # pyainoは正方形のストライドとパディングを前提としているため、先頭の要素を採用します
            pad = pads[0]
            stride = strides[0]
            bias_enabled = bias_name is not None
            
            # Conv2dLayerをインスタンス化します
            layer = Neuron.Conv2dLayer(M, Fh, stride, pad, bias=bias_enabled)
            
            # 畳み込みパラメータ情報を初期化します (config: C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow)
            layer.config = (C, None, None, M, Fh, Fw, stride, stride, pad, None, None)
            layer.parameters.init_parameter()
            
            # 【重要】ONNXの4次元重み [M, C, Fh, Fw] を pyaino用の2次元 [C*Fh*Fw, M] に変換します。
            # リシェイプしたうえで、列優先になるよう転置（.T）を行います。
            w_pyaino = w_onnx.reshape(M, -1).T
            layer.parameters.w = w_pyaino
            
            if bias_enabled:
                layer.parameters.b = b_onnx
                
            pyaino_layers.append(layer)
            
        # --- ReLU 活性化関数の復元 ---
        elif op_type == 'Relu':
            # エクスポート時に埋め込まれた（ノード名に "embedded" が含まれる）ものであれば、
            # 新規レイヤーとして追加するのではなく、直前のレイヤーの後処理(postphase.activator)としてドロップインします。
            if "embedded" in node.name:
                if len(pyaino_layers) > 0:
                    prev_layer = pyaino_layers[-1]
                    if hasattr(prev_layer, 'postphase'):
                        prev_layer.postphase.activator = Activators.ReLU()
            else:
                # 独立した計算ノードとして構築されている場合は、独立したレイヤーとして追加します
                pyaino_layers.append(Activators.ReLU())
                
        # --- LeakyReLU 活性化関数の復元 ---
        elif op_type == 'LeakyRelu':
            alpha = 0.01
            for attr in node.attribute:
                if attr.name == 'alpha':
                    alpha = attr.f
            
            if "embedded" in node.name:
                if len(pyaino_layers) > 0:
                    prev_layer = pyaino_layers[-1]
                    if hasattr(prev_layer, 'postphase'):
                        # 【重要】pyaino側のLReLUの引数 'c'（負の傾き）にONNXの 'alpha' を設定します。
                        prev_layer.postphase.activator = Activators.LReLU(c=alpha)
            else:
                pyaino_layers.append(Activators.LReLU(c=alpha))
                
        # --- Sigmoid 活性化関数の復元 ---
        elif op_type == 'Sigmoid':
            if "embedded" in node.name:
                if len(pyaino_layers) > 0:
                    prev_layer = pyaino_layers[-1]
                    if hasattr(prev_layer, 'postphase'):
                        prev_layer.postphase.activator = Activators.Sigmoid()
            else:
                pyaino_layers.append(Activators.Sigmoid())
                
        # --- Tanh 活性化関数の復元 ---
        elif op_type == 'Tanh':
            if "embedded" in node.name:
                if len(pyaino_layers) > 0:
                    prev_layer = pyaino_layers[-1]
                    if hasattr(prev_layer, 'postphase'):
                        prev_layer.postphase.activator = Activators.Tanh()
            else:
                pyaino_layers.append(Activators.Tanh())
                
        # --- Softmax 活性化関数の復元 ---
        elif op_type == 'Softmax':
            if "embedded" in node.name:
                if len(pyaino_layers) > 0:
                    prev_layer = pyaino_layers[-1]
                    if hasattr(prev_layer, 'postphase'):
                        prev_layer.postphase.activator = Activators.Softmax()
            else:
                pyaino_layers.append(Activators.Softmax())
                
        else:
            print(f"サポートされていない演算ノード '{node.name}' (型: {op_type}) をスキップします。")
            
    # 再構築された全レイヤーをまとめて Sequential モデルにします
    sequential_model = Neuron.Sequential(*pyaino_layers)
    return sequential_model

# --- 単体実行時の検証テストセクション ---
if __name__ == '__main__':
    print("--------------------------------------------------")
    print("ONNX -> pyaino 直接インポート検証テストを実行中...")
    print("--------------------------------------------------")
    
    # 変換対象となるONNXファイルを指定 (pyaino_to_onnx.pyによって生成されたファイル)
    onnx_path = "verification_model.onnx"
    
    if not os.path.exists(onnx_path):
        print(f"[ERROR] {onnx_path} が見つかりません。まず pyaino_to_onnx.py を実行してください。")
        sys.exit(1)
        
    # ONNXからpyainoモデルを復元
    restored_model = import_onnx_to_pyaino(onnx_path)
    
    # 復元されたモデルの構造を確認
    print("\n復元された pyaino モデルの構造概要:")
    restored_model.summary()
    
    # 検証用の固定の入力データを用意 (再現性のためシードを固定)
    np.random.seed(42)
    test_x = np.random.randn(4, 10).astype(np.float32)
    
    # 復元した pyaino モデルの推論を実行
    restored_output = restored_model.forward(test_x)
    print("\n復元された pyaino モデルの出力結果:\n", restored_output)
    
    # 元のONNXファイルをONNX Runtimeで読み込み基準出力を計算
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_outputs = ort_session.run(None, {input_name: test_x})
    ort_output = ort_outputs[0]
    
    print("\nONNX Runtime の基準出力結果:\n", ort_output)
    
    # 2つの出力の絶対誤差を計算
    diff = np.abs(restored_output - ort_output)
    max_difference = np.max(diff)
    print(f"\n復元モデル と ONNX Runtime 間の最大絶対誤差: {max_difference:.2e}")
    
    # 誤差が 1e-6 以下ならテスト合格
    if max_difference < 1e-6:
        print("[PASS] ONNX -> pyaino 復元検証成功！ 出力結果は完全に一致しています。")
    else:
        print("[FAIL] 復元モデルとONNX計算結果の間に不一致が検出されました。")
        sys.exit(1)
