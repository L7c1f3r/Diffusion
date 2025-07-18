from flask import Flask, request, jsonify, send_from_directory
import os
from all_in_one_scripts import run_pipeline
from werkzeug.utils import secure_filename


app = Flask(__name__)
UPLOAD_FOLDER = './input_images'
STYLE_FOLDER = './style_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STYLE_FOLDER, exist_ok=True)
@app.route('/outputs/<filename>')
def get_output_image(filename):
    return send_from_directory('outputs', filename)

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
        "spring": "图片中的景色是温暖的春天，万物复苏，柳树吐芽，花朵盛开，绿草如茵，河流解冻，空气清新，色彩明亮且充满生机。",
        "summer": "图片中的景色是炎热的夏天，阳光强烈，树木繁茂，绿荫如盖，湖水碧绿，天空湛蓝，蝉鸣阵阵，色彩鲜艳且富有活力。",
        "autumn": "图片中的景色是丰收的秋天，稻田金黄，落叶缤纷，果实累累，天空高远，微风凉爽，色彩温暖且层次丰富。",
        "winter": "图片中的景色是寒冷的冬天，白雪皑皑，覆盖着厚厚的积雪，湖面结冰，天空清澈，色彩冷峻且宁静，大地一片银装素裹，气氛安静祥和。"
    }
    style_id_map = {
        "watercolor": "104",
        "realistic": "201",
        "artistic": "134",
        "cartoon": "116",
    }
    style_id = style_id_map.get(style_id, "201")
    prompt ="你需要严格地遵循景色要求: " + prompt_map.get(season, "图片中的场景是春天")
    print(f"prompt: {prompt}")
    print(f"style_id: {style_id}")


    # 调用你的函数
    try:
        print("try to request tecent clouds")
        output_path = run_pipeline(input_path, prompt, strength, style_id)
        print(f"output_path: {output_path}")
        return jsonify({"status": "success", "generated_image_url": f"http://localhost:3060/outputs/{os.path.basename(output_path)}"})
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    app.run(host='0.0.0.0', port=3060)
