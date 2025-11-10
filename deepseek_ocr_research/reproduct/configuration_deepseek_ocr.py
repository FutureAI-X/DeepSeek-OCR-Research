from transformers import Qwen3Config

class DeepseekOCRConfig(Qwen3Config):
    """DeepSeekOCRConfig 配置类
    
    1. 官方的代码中继承了 DeepseekV2Config
    2. 此处我们继承 Qwen3COnfig
    """
    model_type = "DeepSeekOCR"