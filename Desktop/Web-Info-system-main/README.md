# 关于 Web-Info-system

## 项目介绍

Web-Info-system 是一个小型的前端基于原生 HTML5,CSS3 及 Javascript,Jquery 和 Bootstrap,后端基于 Flask 及图像处理库 Pillow 构建的 Web 学生信息管理系统。该系统允许用户查看、编辑和删除学生信息，提供了直观的用户界面和后端数据处理逻辑。

## 主要特性

- 登录注册功能,包括防止 SQL 注入,提供验证码验证功能及刷新功能
- 查看全部学生信息
- 动态修改学生信息
- 删除学生记录
- 信息更新后自动刷新列表及左侧学生信息列表

## 技术栈

- **前端：** HTML5, CSS3, Bootstrap, JavaScript, jQuery
- **后端：** Flask, SQLAlchemy
- **数据库：** SQLite

## 如何运行

### 克隆仓库

你可以通过 git bash 克隆至你的文件夹中:

```sh
git clone https://github.com/ShibaCodingLife/web-info-system.git
```

或者手动访问<https://github.com/ShibaCodingLife/Web-Info-system> 来获取。

### 安装依赖

确保您的系统已安装以下依赖,请确保在安装依赖之前激活您的 Python 虚拟环境：

- Python
- poetry
  - Flask
  - Werkzeug
  - PyYAML
  - pycryptodome
  - Pydantic
- npm
  - argon2-browser

您可以运行以下命令来安装所需的依赖包:

```sh
pip install -U poetry
poetry install --no-root
npm install --prefix ./static argon2-browser
```

使用以下命令运行
```sh
poetry run flask run
```

### 使用 Pillow 进行图像处理

该项目使用 Pillow 库来进行图像处理，这包括图像绘制、添加文本、应用滤镜等功能。

首先，您需要安装 Pillow：

```sh
pip install Pillow
```

在进行简单的验证:

```sh
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# 打开一个图像文件
image = Image.open('example.jpg')

# 在图像上绘制一个矩形
draw = ImageDraw.Draw(image)
draw.rectangle((70, 50, 270, 200), outline='red', fill=None)

# 添加文本到图像上
font = ImageFont.truetype('arial.ttf', 36)
draw.text((70, 250), 'Hello World', fill='blue', font=font)

# 应用滤镜
blurred_image = image.filter(ImageFilter.BLUR)

# 保存修改后的图像
blurred_image.save('output.jpg')
```

如果调试正确就说明配置没有问题了。

### CDN 导入

在前端中通过 link 方式导入了 google icon,bootstrap 的 css 和 js 以及 jquery,旨在减小项目体积方便使用,注意查看您的网络状态,确认这些包体和组件已经成功导入。

### 运行项目并调试

## 版本与技术说明

由于是原生 JS,功能实现较为麻烦,目前的项目不是很成熟,包括前端界面以及后端的功能并没有达到预期效果,需要继续进行完善。
