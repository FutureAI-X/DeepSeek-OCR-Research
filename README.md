# DeepSeek-OCR Research 

## 一 环境说明
torch安装
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

## 二 复用的模型权重
### 2.1 SAM-base
#### 2.1.1 相关资料
- [HF模型权重](https://huggingface.co/facebook/sam-vit-base)
- [MS模型权重](https://modelscope.cn/models/facebook/sam-vit-base)
- [GitHub链接(含pth格式权重)](https://github.com/facebookresearch/segment-anything)
- [官网](https://segment-anything.com/)
- [论文](https://arxiv.org/abs/2304.02643)
#### 2.1.2 模型权重下载
下载命令
```
modelscope download --model facebook/sam-vit-base --local_dir /mnt/workspace/models/facebook/sam-vit-base
```
#### 2.1.3 权重兼容情况
详细分析记录见wweight_handle目录
| 模型权重格式 | 是否兼容 |
| --- | --- |
| .pth | √ |
| .bin | X |
| .safetensor | X |
#### 2.1.4 模型权重转换
- 只取 ```image_encoder.``` 开头的key
- 去除 ```image_encoder.``` 前缀
- shape 已确认一致

### 2.2 CLIP-large
#### 2.2.1 相关资料
- [HF模型权重](https://huggingface.co/openai/clip-vit-large-patch14)
- [MS模型权重](https://modelscope.cn/models/AI-ModelScope/clip-vit-large-patch14)
- [GitHub链接](https://github.com/OpenAI/CLIP)
- [论文](https://arxiv.org/abs/2103.00020)
#### 2.2.2 模型权重下载
下载命令
```
modelscope download --model AI-ModelScope/clip-vit-large-patch14 --local_dir /mnt/workspace/models/openai/clip-vit-large-patch14
```
```
modelscope download --model AI-ModelScope/clip-vit-large-patch14-336 --local_dir /mnt/workspace/models/openai/clip-vit-large-patch14-336
```
#### 2.2.3 权重兼容情况
详细分析记录见weight_analysis目录
| 模型权重格式 | 是否兼容 |
| --- | --- |
| .bin | √ |
| .safetensor | √ |
#### 2.2.4 模型权重转换
- 只取 ```vision_model.```开头的key
- 去除 ```vision_model.``` 前缀
- 去除后的key如果 ```encoder.```开头则替换为```transformer.```开头
- qkv矩阵合并
    - layerid 从0至23

for k in sorted(state_dict_clip_bin_processed.keys()):
    print(f"{k}, {state_dict_clip_bin_processed[k].shape}")
    
原始Key
```
transformer.layers.0.self_attn.q_proj.bias, torch.Size([1024])
transformer.layers.0.self_attn.q_proj.weight, torch.Size([1024, 1024])
transformer.layers.0.self_attn.k_proj.bias, torch.Size([1024])
transformer.layers.0.self_attn.k_proj.weight, torch.Size([1024, 1024])
transformer.layers.0.self_attn.v_proj.bias, torch.Size([1024])
transformer.layers.0.self_attn.v_proj.weight, torch.Size([1024, 1024])
```
合并后Key
```
transformer.layers.0.self_attn.qkv_proj.bias, torch.Size([3072])
transformer.layers.0.self_attn.qkv_proj.weight, torch.Size([3072, 1024])
```