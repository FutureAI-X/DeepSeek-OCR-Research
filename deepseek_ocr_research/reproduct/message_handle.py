from typing import List
from conversation import get_conv_template
from addict import Dict

def build_conversation(prompt: str, image_file:str):
    """构建消息列表"""
    return [
                {
                    "role": "<|User|>",
                    "content": f'{prompt}',
                    "images": [f'{image_file}'],
                },
                {
                    "role": "<|Assistant|>", 
                    "content": ""
                },
            ]

def apply_sft_template(conversations: List[Dict[str, str]], sft_format: str = "deepseek", system_prompt: str = ""):
    """将SFT模板应用到conversations中

    Args:
        conversations: 消息列表
        sft_format: 要使用的 SFT 模板
        system_prompt: 在SFT模板中的系统提示

    Returns:
        sft_prompt 格式化后的文本
    
    """
    conv = get_conv_template(sft_format)
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    sft_prompt = conv.get_prompt().strip()

    return sft_prompt