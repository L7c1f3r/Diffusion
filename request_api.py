from flask import Flask, request, jsonify
import os
from all_in_one_scripts import run_pipeline
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = './input_images'
STYLE_FOLDER = './style_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STYLE_FOLDER, exist_ok=True)

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline_api():
    # 处理文件上传
    input_image_file = request.files.get('input_image')
    style_files = request.files.getlist('style_images')
    experiment_name = request.form.get('experiment_name', 'test_experiment')

    # 保存输入图片
    if input_image_file:
        input_image_path = os.path.join(UPLOAD_FOLDER, secure_filename(input_image_file.filename))
        input_image_file.save(input_image_path)
    else:
        return jsonify({'status': 'error', 'msg': 'No input image uploaded'}), 400

    # 保存风格图片
    for style_file in style_files:
        style_path = os.path.join(STYLE_FOLDER, secure_filename(style_file.filename))
        style_file.save(style_path)

    try:
        result = run_pipeline(
            input_image=input_image_path,
            experiment_name=experiment_name,
            style_images_dir=STYLE_FOLDER
        )
        return jsonify({
            'status': 'success',
            'output_dir': result['output_dir'],
            'exp_output_root': result['exp_output_root'],
            'content_images_dir': result['content_images_dir'],
            'precomputed_features_dir': result['precomputed_features_dir']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'msg': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
