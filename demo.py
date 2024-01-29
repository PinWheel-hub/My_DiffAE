import lmdb
from PIL import Image
from io import BytesIO

# 打开 LMDB 环境
env = lmdb.open('datasets/ffhq256.lmdb', readonly=True)

# 创建一个事务
with env.begin() as txn:
    # 从数据库中读取图像数据
    image_data = txn.get(b'256-00198')

    # 将二进制数据转换为字节流
    image_stream = BytesIO(image_data)

    # 使用 PIL 读取图像
    try:
        image = Image.open(image_stream)
        image.show()
    except IOError:
        print("无法读取图像")
