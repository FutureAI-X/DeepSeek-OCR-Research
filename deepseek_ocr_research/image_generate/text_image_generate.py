from PIL import Image, ImageDraw, ImageFont

# 导入需要在图片上显示的文本：此处是《滕王阁序》
from text_tengwanggexu import tengwanggexu

def wrap_text(text, font, max_width):
    """
    对单个段落进行自动换行
    返回该段落的行列表
    """
    lines = []
    current_line = ""

    for char in text:
        test_line = current_line + char
        bbox = font.getbbox(test_line)
        line_width = bbox[2] - bbox[0]  # right - left

        if line_width <= max_width:
            current_line += char
        else:
            if current_line:
                lines.append(current_line)
            current_line = char  # 新行从当前字符开始

    if current_line:
        lines.append(current_line)
    return lines


# ---------------- 配置参数 ----------------
width, height = 512, 512            # 图片分辨率（可调）
background_color = "white"          # 背景颜色
text_color = "black"                # 文字颜色    
font_size = 32                      # 文字大小
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
margin = 40                         # 边距
line_spacing = 10                   # 行间距
paragraph_spacing = 20              # 段落间距（额外空隙）

text = tengwanggexu.strip()

# -------------------------------------------

# 创建图像
image = Image.new("RGB", (width, height), background_color)
draw = ImageDraw.Draw(image)

# 加载字体
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    print(f"⚠️ 未找到字体: {font_path}")
    font = ImageFont.load_default()
    # 推荐安装中文字体：sudo apt install fonts-noto-cjk

# 计算字体高度和每行所需垂直空间
temp_bbox = font.getbbox("国")
font_height = temp_bbox[3] - temp_bbox[1]
line_height = font_height + line_spacing  # 每行占用的高度

# 最小绘制高度：至少能画一行文字 + 底部 margin
min_required_height = line_height + margin

# 按段落分割
paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

# 开始绘制
y = margin  # 起始 y 坐标

for para in paragraphs:
    # 清理段落内换行
    cleaned_para = para.replace('\n', ' ')
    if not cleaned_para:
        continue

    # 对当前段落换行
    wrapped_lines = wrap_text(cleaned_para, font, width - 2 * margin)

    # 判断段落是否可以整体绘制（可选：若想整段一起判断）
    # 这里我们逐行判断更灵活

    for line in wrapped_lines:
        # ✅ 关键判断：剩余空间是否足够绘制这一行？
        if y + line_height > height - margin:
            print(f"⚠️ 剩余空间不足，停止绘制。当前位置 y={y}, 可用到底部: {height - margin}")
            break  # 跳出当前行循环

        # 绘制文字
        draw.text((margin, y), line, fill=text_color, font=font)
        y += line_height
    else:
        # 如果段落所有行都成功绘制，则加上段落间距
        y += paragraph_spacing
        continue  # 继续下一个段落

    # 如果是因为空间不足跳出的，则终止所有绘制
    print("📌 已达到图像底部，停止渲染后续内容。")
    break

# 保存图片
image.save("tengwangge.png")
print(f"✅ 图片已生成：tengwangge.png")

# 可选：显示图片（需 GUI）
# image.show()