import base64
import io
import os
import json
import types
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.aiart.v20221229 import aiart_client, models
from flask import Flask, request, jsonify, send_from_directory
import os
from dotenv import load_dotenv
from flask_cors import CORS
import time

load_dotenv()
app = Flask(__name__)
CORS(app)

@app.route('/outputs/<filename>')
def get_output_image(filename):
    return send_from_directory('outputs', filename)
def request_tecent_cloud(input_image = None, prompt = "", imput_url = None, style_id = 201, strength = 0.8):
    if input_image is None:
        raise ValueError("input_image cannot be None")
    if input_image is not None:
        input_image = open(input_image, "rb").read()
        input_image = base64.b64encode(input_image).decode('utf-8')
    try:
        cred = credential.Credential(os.getenv("TENCENTCLOUD_SECRET_ID"), os.getenv("TENCENTCLOUD_SECRET_KEY"))
        httpProfile = HttpProfile()
        httpProfile.endpoint = "aiart.tencentcloudapi.com"

        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        client = aiart_client.AiartClient(cred, "ap-guangzhou", clientProfile)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.ImageToImageRequest()
        
        params = {
            "Prompt": prompt,
            "Styles": [
                style_id
            ],
            "Strength": strength,
        }
        if input_image is not None:
            params["InputImage"] = input_image
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个ImageToImageResponse的实例，与请求对象对应
        resp = client.ImageToImage(req)
        resp = resp.to_json_string()
        resp = json.loads(resp)
        img_base64 = resp["ResultImage"]
        img_bytes = base64.b64decode(img_base64)
        now = int(time.time())
        output_path = f"outputs/result_{now}.jpg"
        with open(output_path, "wb") as f:
            f.write(img_bytes)
            print("success write image")
        return output_path
    except TencentCloudSDKException as err:
        print(err)
        return None

@app.route("/api/generate", methods=["POST"])
def generate():
    # 获取参数
    input_image = request.files.get("input_image")
    if input_image is None:
        return jsonify({"status": "error", "error": "input_image is required"})
    os.makedirs("uploads", exist_ok=True)
    input_path = os.path.join("uploads", input_image.filename)
    input_image.save(input_path)
      
    season = request.form.get("season", "spring")
    strength_str = request.form.get("strength", "0.8")
    try:
        strength = float(strength_str) if strength_str else 0.8
    except ValueError:
        strength = 0.8
    style_id = request.form.get("style_id", "Realistic")

    # prompt 映射
    prompt_map = {
        "spring": "图片中的景色是温暖的春天，万物复苏，柳树吐出嫩绿的新芽，樱花、桃花、迎春花等各色花卉盛开，草地上长满了鲜嫩的绿草，灌木丛生机盎然，河流解冻，山坡上点缀着野花，田野湿润，空气清新，天空湛蓝，色彩明亮且充满生机。",
        "summer": "图片中的景色是炎热的夏天，阳光强烈，树木枝繁叶茂，榕树、梧桐、杨树等高大乔木形成浓密的绿荫，草地郁郁葱葱，荷花在池塘中盛开，藤蔓植物攀爬生长，湖水碧绿，天空湛蓝，蝉鸣阵阵，远处有连绵的山峦，田野里庄稼茁壮生长，河流清澈见底，色彩鲜艳且富有活力。",
        "autumn": "图片中的景色是丰收的秋天，稻田金黄，银杏、枫树、槭树等树木叶片变成金黄或红色，落叶缤纷，果园里苹果、柿子、葡萄等果实累累，草地逐渐变黄，天空高远，微风凉爽，山林层林尽染，河岸边芦苇随风摇曳，田野里南瓜和玉米成熟，色彩温暖且层次丰富。",
        "winter": "图片中的景色是寒冷的冬天，白雪皑皑，落叶乔木的枝干上覆盖着厚厚的积雪，灌木和草地被雪埋没，湖面结冰，天空清澈，远处山峦银装素裹，树枝上挂满冰凌，河流冻结，房屋和田野都被雪覆盖，色彩冷峻且宁静，大地一片银装素裹，气氛安静祥和。"
    }
    style_id_map = {
        "watercolor": "104",
        "realistic": "201",
        "artistic": "134",
        "cartoon": "116",
    }
    style_id = style_id_map.get(style_id, "201")
    prompt = prompt_map.get(season, "图片中的场景是春天")
    prompt ="你需要严格地遵循景色要求: " + prompt_map.get(season, "图片中的场景是春天")
    print(f"prompt: {prompt}")
    print(f"style_id: {style_id}")
    print(f"strength: {strength}")

    # 调用你的函数
    try:
        print("try to request tecent clouds")
        output_path = request_tecent_cloud(input_path, prompt, strength, style_id)
        print(f"output_path: {output_path}")
        return jsonify({"status": "success", "generated_image_url": f"http://localhost:3060/outputs/{os.path.basename(output_path)}"})
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    app.run(host='0.0.0.0', port=3060)
    