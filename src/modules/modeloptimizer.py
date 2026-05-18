from pathlib import Path
from typing import Tuple, Dict
import warnings
from collections import defaultdict
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.algorithms import AutoQuantizeGradientSearcher, QuantRecipe
from modelopt.torch.quantization.mode import QuantizeModeRegistry
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader
from modelopt.torch.opt import apply_mode
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
import modelopt.torch.opt as mto

from ..aqt.utils.common import seed_everything, cleanup_memory
from ..utils.global_args import GlobalConfig
from ..modules.quantizer import LLMCompressorQuantizer

SEED = 42
seed_everything(SEED)


class ModelOptimizerQuantizer():
    def __init__(self, config: GlobalConfig):
        self.model_path = config.raw_config["base_model_path"]
        self.quant_config = config.raw_config['quantization']
        self.data_path = self.quant_config["calib_data_path"]
        self.base_dir = Path(config.raw_config["workspace"]["base_dir"])
        self.export_path = self.base_dir / config.raw_config["workspace"]["best_weights_dir"]
        self.disabled_layers = config.raw_config["strategy"]["initial_fallback_layers"] + config.raw_config["disable_names"]
        self.device = self.quant_config['device']
        self.visible_devices = self.quant_config['visible_devices']

    
    def run(self):
        """
        执行完整的量化流程。
        """
        best_recipe = self._get_best_config()["recipe"]
        w8a8_default = []
        for key, value in best_recipe.items():
            layer_name = "*" + ".".join(key.split(".")[:-1]) + "*"
            if "INT8_DEFAULT_CFG" in str(value):
                w8a8_default.append(layer_name)

        hybrid_quant_schema, hybrid_quant_schema_re = self._generate_schema(w8a8_default)
        hybrid_quant_schema_path = self.base_dir / "hybrid_quant_schema.json"
        hybrid_quant_schema_re_path = self.base_dir / "hybrid_quant_schema_re.json"

        with open(hybrid_quant_schema_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_quant_schema, f, indent=4)
        with open(hybrid_quant_schema_re_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_quant_schema_re, f, indent=4)
        
        self._compress_model(hybrid_quant_schema_path, hybrid_quant_schema_re_path)
    
    def _generate_schema(self, w8a8_default):
        hybrid_quant_schema = {
            "w8a8_default": {"inlcude": w8a8_default, "exclude": []},
        }
        def parse_layer(layer_pattern):
            # 去掉首尾的 *
            layer = layer_pattern.strip('*')
            # 提取层号和各部分名称
            parts = layer.split('.')
            # 找到 layers.数字 的位置
            layer_idx = None
            for i, part in enumerate(parts):
                if part == 'layers' and i+1 < len(parts):
                    layer_idx = parts[i+1]
                    break
            if layer_idx is None:
                return None
            # 提取子模块类型和名称
            submodule = '.'.join(parts[parts.index('layers')+2:])
            return {'layer_idx': layer_idx, 'submodule': submodule}
        mapping = defaultdict(dict)

        # 处理 w8a8 层
        for layer in w8a8_default:
            parsed = parse_layer(layer)
            if parsed:
                key = parsed['submodule']
                mapping[key]["w8a8_default"] = mapping[key].get("w8a8_default", []) + [layer]

        # 现在运行你提供的逻辑
        def _sort_mapping(mapping):
            # 简单排序，保证确定性
            return {k: dict(sorted(v.items())) for k, v in sorted(mapping.items())}

        def _get_re_format(layer_names):
            # 将层名列表转换为正则表达式
            if not layer_names:
                return []
            # 提取层号并去重
            layer_nums = set()
            base_pattern = None
            for name in layer_names:
                name_clean = name.strip('*')
                parts = name_clean.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i+1 < len(parts):
                        layer_nums.add(parts[i+1])
                        if base_pattern is None:
                            # 构建基础模式，用 {layer} 代替层号
                            parts[i+1] = "{layer}"
                            base_pattern = ".*" + ".".join(parts) + ".*"
                        break
            
            if base_pattern:
                layer_nums_sorted = sorted(layer_nums, key=int)
                layer_pattern = "|".join(layer_nums_sorted)
                re_pattern = base_pattern.replace("{layer}", f"({layer_pattern})")
                return [f"re:{re_pattern}"]
            return []

        mapping = _sort_mapping(mapping)
        quant_schemas_all = set()
        for v in mapping.values():
            quant_schemas_all.update(v.keys())


        hybrid_quant_schema_re = {k: [] for k in quant_schemas_all}
        for quant_schema in quant_schemas_all:
            for pattern in mapping.keys():
                if quant_schema in mapping[pattern]:
                    for re_names in _get_re_format(mapping[pattern][quant_schema]):
                        hybrid_quant_schema_re[quant_schema].append(re_names)

        # 移除 float（如果有）
        if "float" in quant_schemas_all:
            del hybrid_quant_schema_re["float"]
        
        return hybrid_quant_schema, hybrid_quant_schema_re

    def _compress_model(self, hybrid_quant_schema_path, hybrid_quant_schema_re_path):
        """
        压缩模型。
        """
        quant_log_path = self.base_dir / "llmcompressor.log"
        quant_config_path = self.base_dir / "generated_llmcompressor_config.py"
        quantizer = LLMCompressorQuantizer(
            quant_config={
                'device': self.device,
                'visible_devices': self.visible_devices,
                'calib_data_path': self.data_path,
                'num_calibration_samples': self.quant_config["calib_samples"],
            },
            base_model_path=self.model_path,
            fallback_layers=self.disabled_layers,
            output_config_path=str(quant_config_path),
            output_weights_path=str(self.export_path),
            hybrid_quant_schema_path=hybrid_quant_schema_path,
            hybrid_quant_schema_re_path=hybrid_quant_schema_re_path,
            quant_log_path=str(quant_log_path),
        )

        quantized_model_path = quantizer.run()
        print(f"Quantized model path: {quantized_model_path}")

    def _get_best_config(self) -> Dict:
        """
        Returns best quantization format.
        """
        model, tokenizer = self._set_up_model_and_tokenizer(self.model_path)
        data_loader = get_dataset_dataloader(
            dataset_name=self.quant_config["calib_data_path"],
            tokenizer=tokenizer,
            batch_size=self.quant_config["batch_size"],
            num_samples=self.quant_config["calib_samples"],
            device=self._normalize_device(self.device, self.visible_devices),
            include_labels=True
        )
        quantization_formats = [mtq.INT8_DEFAULT_CFG]
        processed_quantization_formats = []
        for i, quant_cfg in enumerate(quantization_formats):
            if quant_cfg is None:
                continue

            name = QuantRecipe.get_auto_name_for_config(quant_cfg)
            if name is None:
                name = f"CUSTOM_{i}"
                warnings.warn(
                    f"Received custom quantization formats for search, auto_quantize results may not be optimal. "
                    f"This config will be displayed as {name}"
                )
            processed_quantization_formats.append((quant_cfg, name))

        searcher = AutoQuantizeGradientSearcher()
        model = apply_mode(
            model,
            mode="auto_quantize",
            registry=QuantizeModeRegistry,
        )
        search_config = {
            "quantization_formats": processed_quantization_formats,
            "data_loader": data_loader,
            "forward_step": lambda m, b: m(**b),
            "loss_func": lambda out, batch: out.loss,
            "forward_backward_step": None,
            "num_calib_steps": len(data_loader),
            "num_score_steps": len(data_loader),
            "disabled_layers": self.disabled_layers,
            "verbose": True,
            "checkpoint": None,
        }
        # Disable all quantizers; AutoQuantize will enable the needed ones
        set_quantizer_by_cfg(model, {"*": {"enable": False}})

        result =  searcher.search(model, {"effective_bits": self.quant_config["effective_bits"]}, config=search_config)
        del model
        cleanup_memory()

        return result

    def _normalize_device(self, device: str, visible_devices: str) -> str:
        if device == "cpu":
            return "cpu"
        visible_devices = visible_devices.split(",")[0].strip()
        return f"{device}:{visible_devices}"
    
    def _set_up_model_and_tokenizer(self, model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Returns model in eval mode and tokenizer.
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=self._normalize_device(self.device, self.visible_devices),
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer

    def export_model(self, model: AutoModelForCausalLM, export_path: Path | str) -> None:
        """
        Export model to HuggingFace format.
        """
        if "npu" in self.device:
            model = model.to("cpu")

        mto.enable_huggingface_checkpointing()
        model.save_pretrained(str(export_path))