import random
import io
import yaml
import logging
# 需要导入的包体,flask,request不用说,render_template用来渲染定位, jsonify是后端传给前端的数据格式以便前端获取其中的信息来做出判断,redirect重定向
from flask import Flask, request, render_template, jsonify, redirect, make_response
from flask_sqlalchemy import SQLAlchemy
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
from base64 import b64encode, b64decode
from Crypto.Cipher import AES
from pydantic import BaseModel
from datetime import datetime, timedelta
from uuid import uuid4
from cachetools import TTLCache


class Config(BaseModel):
    class DBConfig(BaseModel):
        uri: str
        username: str = ""
        password: str = ""
    class CookiesConfig(BaseModel):
        aes_key: str  # Encryption key (must be 16, 24, or 32 bytes long)
        expire_sec: int
    class GreetingConfig(BaseModel):
        max_sessions: int
        expire_sec: int
    
    version: int
    debug: bool
    cookies: CookiesConfig
    greeting: GreetingConfig  # used for login & register
    logging: dict
    db: DBConfig


DEFAULT_CONFIG = Config.model_validate({
    "version": 1,
    "debug": True,
    "cookies": {
        "aes_key": "shiba_is_best&&$@SDU%^%#peropero",
        "expire_sec": 3600 * 24 * 7,  # 7 days
    },
    "greeting": {
        "max_sessions": 2048,
        "expire_sec": 300,
    },
    "logging": {
        "level": "DEBUG",
        "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
    },
    "db": {
        "uri": "sqlite:///shiba.db"
    }
})


class Cookies(BaseModel):
    username: str
    token: str
    expire: datetime


CONFIG_PATH = Path(__file__).parent / "config.yaml"
LOG_PATH = Path(__file__).parent / "app.log"

if not CONFIG_PATH.exists():
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(DEFAULT_CONFIG.model_dump(), f)
with open(CONFIG_PATH) as f:
    config = Config.model_validate(yaml.safe_load(f))

app = Flask(__name__)  # 初始化
app.debug = config.debug
# app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:320421@localhost:3306/sakila"##数据库连接,在终端下载SQLAlchemy后用此命令连接,但有可能会出现mysql认证问题,所以建议再下一个cryptography 包,命令为pip install cryptography
app.config["SQLALCHEMY_DATABASE_URI"] = config.db.uri
db = SQLAlchemy(app)
logging.basicConfig(filename=LOG_PATH, **config.logging)

if app.debug:
    logging.getLogger().addHandler(logging.StreamHandler())

code_map = TTLCache[str, str](
    maxsize=config.greeting.max_sessions,
    ttl=config.greeting.expire_sec)

class Student_base_info(db.Model):  # 构造对象
    __tablename__ = "teacher_info"  # 定义是mysql中的哪一个表
    id = db.Column(db.Integer, primary_key=True)
    # 数据条定义,必须定义主键,否则flask会因在sqlite中找不到匹配模块而报错
    name = db.Column(db.String(100))
    password = db.Column(db.String(100))


class Teacher_stu_info(db.Model):
    __tablename__ = "teacher-stu"
    id = db.Column(db.Integer, primary_key=True)
    teachername = db.Column(db.String(100))
    studentname = db.Column(db.String(100))
    studentnumber = db.Column(db.String(100))
    studentsex = db.Column(db.String(100))
    studentage = db.Column(db.String(100))
    studentorigin = db.Column(db.String(100))
    studentsdept = db.Column(db.String(100))


def encrypt_cookies(cookies: Cookies) -> str:
    cipher = AES.new(config.cookies.aes_key.encode(), AES.MODE_GCM)
    encrypted, digest = cipher.encrypt_and_digest(cookies.model_dump_json().encode())
    data = len(digest).to_bytes(1, "little") + digest + cipher.nonce + encrypted
    return b64encode(data).decode()

def decrypt_cookies(b64: str) -> Cookies:
    data = b64decode(b64)
    d_len = data[0]
    digest = data[1:1+d_len]
    cipher = AES.new(config.cookies.aes_key.encode(),
                     AES.MODE_GCM, nonce=data[d_len + 1:d_len + 17])
    decrypted: bytes = cipher.decrypt_and_verify(data[d_len + 17:], digest)
    cookies = Cookies.model_validate_json(decrypted.decode())
    return cookies

def validate_cookies(cookies: str | Cookies) -> bool:
    if isinstance(cookies, str):
        try:
            cookies = decrypt_cookies(cookies)
        except ValueError:
            return False

    teacher_info = Student_base_info.query.filter_by(
        name=cookies.username, password=cookies.token).first()

    return teacher_info and cookies.expire > datetime.utcnow()

@app.route('/')
def home():
    cookies = request.cookies.get("cookies")
    if not cookies or not validate_cookies(cookies):
        return redirect("/login")
    return redirect("/new.html")
    

# 显示登录页面，仅允许GET请求


@app.route("/login", methods=["GET"])
def login():
    captcha_code = generate_random_code()
    image = generate_image(captcha_code)
    b = io.BytesIO()
    image.save(b, "jpeg")
    image_path = f"data:image/jpeg;base64,{b64encode(b.getvalue()).decode()}"
    res =  make_response(render_template("login.html", image_path=image_path))
    session_id = str(uuid4())
    res.set_cookie("session_id", session_id, httponly=True)
    code_map[session_id] = captcha_code.lower()
    return res


@app.route("/login_get", methods=["POST"])  # 允许处理前端的post请求
def login_get():
    name = request.form.get("name")
    password = request.form.get("password")
    code = request.form.get("code")
    session_id = request.cookies.get("session_id")

    if not name or not password or not code:
        return jsonify({"success": False, "code": 1, "info": "信息不完整"})
    
    captcha_code = code_map.get(session_id)
    code = code.lower()

    if not captcha_code:
        return jsonify({"success": False, "code": 2, "info": "验证码过期"})

    teacher_info = Student_base_info.query.filter_by(
        name=name, password=password).first()  # 查询过滤器

    logging.debug("Login attempt : name: %s, password: %s", name, password)
    if teacher_info:
        if code == captcha_code:
            res = jsonify({"success": True, "info": "正确"})  # 封装信息并返回
            cookies = Cookies(
                username=name,
                token=password,
                expire=datetime.utcnow() + timedelta(seconds=config.cookies.expire_sec))
            res.set_cookie("cookies", encrypt_cookies(cookies), httponly=True)
            return res
        else:
            return jsonify({"success": False, "code": 3, "info": "验证码错误"})
    else:
        return jsonify({"success": False, "code": 4, "info": "学生信息错误"})


@app.route("/register_get", methods=["POST"])
def register_get():
    name = request.form.get("name")
    password = request.form.get("password")
    code = request.form.get("code")
    session_id = request.cookies.get("session_id")

    if not name or not password or not code:
        return jsonify({"success": False, "code": 1, "info": "信息不完整"})
    
    captcha_code = code_map.get(session_id)
    code = code.lower()

    if not captcha_code:
        return jsonify({"success": False, "code": 2, "info": "验证码过期"})

    teacher_info = Student_base_info.query.filter_by(name=name).first()
    if teacher_info:
        return jsonify({"success": False, "code": 5, "info": "此用户已经存在"})
    if code != captcha_code:
        return jsonify({"success": False, "code": 3, "info": "验证码错误"})
    else:
        new_student = Student_base_info(name=name, password=password)
        db.session.add(new_student)
        db.session.commit()
    return jsonify({"success": True, "info": "注册成功"})


def generate_random_code(length=4):
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    code = ''.join(random.choice(characters) for _ in range(length))
    return code.lower()


def generate_image(code):
    image = Image.new("RGB", (150, 60), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 使用艺术字体
    with open("static/arial.ttf", "rb") as f:
        # Well shit, I don't have arial.ttf on local OS
        font = ImageFont.truetype(f, 40)

    for i in range(4):
        draw.text((10 + i * 30, 10), code[i], font=font, fill=(0, 0, 0))

    for _ in range(20):
        x1 = random.randint(0, 150)
        y1 = random.randint(0, 60)
        x2 = random.randint(0, 150)
        y2 = random.randint(0, 60)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0))

    for _ in range(30):
        x = random.randint(0, 150)
        y = random.randint(0, 60)
        draw.rectangle([x, y, x + 3, y + 3], fill=(0, 0, 0))

    for _ in range(500):
        x = random.randint(0, 150)
        y = random.randint(0, 60)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        draw.point((x, y), fill=(r, g, b))

    image = image.filter(ImageFilter.SMOOTH_MORE)
    return image


@app.route("/change_captcha")
def change_captcha():
    session_id = request.cookies.get("session_id")
    if not session_id:
        return jsonify({"success": False, "code": -1, "info": "未知错误, 请刷新界面"})
    code = generate_random_code()
    image = generate_image(code)
    code_map[session_id] = code
    b = io.BytesIO()
    image.save(b, "jpeg")
    image_path = f"data:image/jpeg;base64,{b64encode(b.getvalue()).decode()}"
    return jsonify({"success": True, "image_path": image_path})


@app.route("/new.html")  # 渲染此页面并查询所有带过的学生名单
def new_page():
    cookies = request.cookies.get("cookies")
    if not cookies or not validate_cookies(cookies):
        return redirect("/login")

    cookies = decrypt_cookies(cookies)
    name = cookies.username
    students = Teacher_stu_info.query.filter_by(teachername=name).all()
    return render_template("new.html", name=name, students=students)


@app.route("/login.html")  # 重定向,更新验证码,并更新全局变量code_num
def redirect_login():
    return redirect("/login")

@app.route("/register.html")  # 同样的更新
def redirect_register():
    return redirect("/register")

@app.route("/register")  # 同样的更新
def register_html():
    captcha_code = generate_random_code()
    image = generate_image(captcha_code)
    b = io.BytesIO()
    image.save(b, "jpeg")
    image_path = f"data:image/jpeg;base64,{b64encode(b.getvalue()).decode()}"
    res =  make_response(render_template("register.html", image_path=image_path))
    session_id = str(uuid4())
    res.set_cookie("session_id", session_id, httponly=True)
    code_map[session_id] = captcha_code.lower()
    return res


@app.route("/all_get", methods=["POST"])  # 前端点击获取所有关于老师学生信息并渲染至表格中
def get_info():
    cookies = request.cookies.get("cookies")
    if not cookies or not validate_cookies(cookies):
        return redirect("/login")
    cookies = decrypt_cookies(cookies)
    name = cookies.username
    logging.debug("Get all info : name: %s", name)
    students = Teacher_stu_info.query.filter_by(teachername=name).all()
    student_list = [{"name": student.studentname, "number": student.studentnumber, "sex": student.studentsex,
                     "age": student.studentage, "origin": student.studentorigin, "sdept": student.studentsdept} for student in students]  # 用你的实际字段替换
    return jsonify({"students": student_list})  # 以json格式返回


@app.route("/delete_student", methods=["POST"])  # 删除当前行学生
def delete_student():
    cookies = request.cookies.get("cookies")
    if not cookies or not validate_cookies(cookies):
        return redirect("/login")
    student_number = request.form.get("student_number")
    # 在数据库中查找该学号的学生
    student = Teacher_stu_info.query.filter_by(
        studentnumber=student_number).first()
    if student:
        db.session.delete(student)  # 删除学生记录
        db.session.commit()  # 提交更改
        # 将名字交给前端用于删除菜单左侧名单
        return jsonify({"success": True, "studentName": student.studentname})
    else:
        return jsonify({"success": False, "code": 100, "error": "删除错误"})


@app.route("/update_student/<student_id>", methods=["POST"])
def update_student(student_id):
    cookies = request.cookies.get("cookies")
    if not cookies or not validate_cookies(cookies):
        return redirect("/login")

    data = request.get_json()  # 获取json文件
    student = Teacher_stu_info.query.filter_by(
        studentnumber=student_id).first()
    # print(student_id)
    # print(data["number"], data["age"], data["address"], data["sdept"])
    # print(student.studentnumber, student.studentage)
    logging.debug("Update student : student_id: %s, data: %s",
                  student_id, data)
    if student:
        student.studentnumber = data["number"]
        student.studentage = data["age"]
        student.studentorigin = data["address"]
        student.studentsdept = data["sdept"]
        db.session.commit()
        return jsonify({"success": True})
    else:
        return jsonify({"success": False}), 404
    
@app.route('/search-by-name',methods=['POST'])#通过学生姓名查询
def search_by_name():
     name = request.form.get('name')
     students=Teacher_stu_info.query.filter_by(studentname=name).all()
     if students:
        student_list = [{'name': student.studentname,'number':student.studentnumber,'sex':student.studentsex,'age':student.studentage,'origin':student.studentorigin,'sdept':student.studentsdept} for student in students]  # 用你的实际字段替换
        return jsonify({'success': True, 'student': student_list})
     else:
        return jsonify({'success':False})


if __name__ == "__main__":
    app.run()
