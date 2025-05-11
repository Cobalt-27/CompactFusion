from PIL import Image
import os

def concatenate_images_vertical(image_paths, output_path):
    """
    将一系列图片垂直拼接成一张图片。

    Args:
        image_paths (list): 包含要拼接的图片文件路径的列表。
        output_path (str): 保存拼接后图片的路径。
    """
    if not image_paths:
        print("没有提供任何图片路径。")
        return

    images = [Image.open(path) for path in image_paths if os.path.exists(path)]
    if not images:
        print("找不到提供的任何图片文件。")
        return

    # 获取所有图片的宽度，并取最宽的作为输出图片的宽度
    widths = [img.width for img in images]
    max_width = max(widths)

    # 计算拼接后图片的总高度
    total_height = sum(img.height for img in images)

    # 创建一个新的空白图片
    new_image = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for img in images:
        # 如果图片宽度小于最大宽度，则需要创建一个新的图片并居中粘贴
        if img.width < max_width:
            temp_img = Image.new('RGB', (max_width, img.height), (0, 0, 0))  # 创建黑色背景
            x_offset = (max_width - img.width) // 2
            temp_img.paste(img, (x_offset, 0))
            new_image.paste(temp_img, (0, y_offset))
        else:
            new_image.paste(img, (0, y_offset))
        y_offset += img.height

    try:
        new_image.save(output_path)
        print(f"已将图片拼接并保存到: {output_path}")
    except Exception as e:
        print(f"保存图片时发生错误: {e}")

if __name__ == "__main__":
    image_files = [
        "keyframes/keyframe_1.jpg",
        "keyframes/keyframe_2.jpg",
        "keyframes/keyframe_3.jpg",
        "keyframes/keyframe_4.jpg",
        "keyframes/keyframe_5.jpg",
        "keyframes/keyframe_6.jpg",
    ]  # 替换为你的实际图片路径列表
    output_file = "vertical_stack.jpg"  # 替换为你想要保存的拼接后图片文件名

    concatenate_images_vertical(image_files, output_file)